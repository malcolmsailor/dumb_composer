from __future__ import annotations

import copy
import logging
import os
import re
import typing as t
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from itertools import chain, count, cycle, repeat
from types import MappingProxyType

import music21
from cache_lib import cacher

from dumb_composer.constants import speller_pcs, unspeller_pcs
from dumb_composer.pitch_utils.aliases import Fifth, Root, Seventh, Third
from dumb_composer.pitch_utils.consonances import pcs_consonant
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.music21_handler import parse_rntxt
from dumb_composer.pitch_utils.pcs import get_pc_complement, pitch_class_among_pitches
from dumb_composer.pitch_utils.spacings import (
    RangeConstraints,
    SpacingConstraints,
    yield_spacings,
)
from dumb_composer.pitch_utils.types import (
    TIME_TYPE,
    BassFactor,
    ChordFactor,
    ChromaticInterval,
    Pitch,
    PitchClass,
    PitchOrPitchClass,
    RNToken,
    ScalarInterval,
    ScaleDegree,
    TimeStamp,
    VoiceCount,
)
from dumb_composer.time import Meter
from dumb_composer.utils.iterables import yield_sample_from_sequence_of_iters

LOGGER = logging.getLogger(__name__)

# TODO: (Malcolm 2023-07-19) discourage doubling resolution of suspension
# TODO: (Malcolm 2023-07-19) customize penalty per chord for omitting specific chord factors
#   for example, omitting 5th of tonic triad should bear little penalty; same for dominant triad


class Allow(Enum):
    # For omissions,
    #   NO means pitch cannot be omitted
    #   YES means pitch can be omitted
    #   ONLY means pitch *must* be omitted
    NO = auto()
    YES = auto()
    ONLY = auto()


class Tendency(Enum):
    NONE = auto()
    UP = auto()
    DOWN = auto()


RNTokenWithoutFigure = str

AbstractChordTendencies = t.Mapping[ChordFactor, Tendency]
ConcreteChordTendencies = t.Mapping[BassFactor, Tendency]


class Inflection(Enum):
    EITHER = auto()
    NONE = auto()
    UP = auto()
    DOWN = auto()


@dataclass
class Resolution:
    by: ChromaticInterval
    to: PitchClass


# TODO: (Malcolm 2023-08-16) treat sevenths on chords other than dominants as
#   presumptive suspensions when the preparation occurs in the preceding chord
#   (This will likely take a lot of work).

TENDENCIES: t.Mapping[RNTokenWithoutFigure, AbstractChordTendencies] = MappingProxyType(
    {
        # V: 3rd: up; 7th: down
        "V": {Third: Tendency.UP, Seventh: Tendency.DOWN},
        "viio": {Root: Tendency.UP, Fifth: Tendency.DOWN, Seventh: Tendency.DOWN},
        # #viio seems to be a mistake or at least an idiosyncracy in some of the
        # annotations
        "#viio": {Root: Tendency.UP, Fifth: Tendency.DOWN, Seventh: Tendency.DOWN},
        "Ger": {Root: Tendency.UP, Third: Tendency.DOWN},
        "Fr": {Third: Tendency.UP, Fifth: Tendency.DOWN},
        "iv": {Third: Tendency.DOWN},
        # TODO: (Malcolm 2023-08-16) do we need to set the seventh tendency explicitly
        # for every chord?
        "ii": {Seventh: Tendency.DOWN},
        # To get the correct tendencies for the cadential 64 chord we need
        #   to index into it as I. Sorry Schenkerians!
        "Cad": {Root: Tendency.DOWN, Third: Tendency.DOWN},
        # TODO: (Malcolm 2023-07-12) test N
        "N": {Root: Tendency.DOWN},
    }
)

DEFAULT_CHORD_FACTOR_SUSPENSION_WEIGHT = 0.5
SUSPENSION_WEIGHTS_BY_SCALAR_INTERVALS: t.Mapping[
    tuple[ScalarInterval, ...], t.Mapping[ChordFactor, float]
] = {
    # TRIADS
    # 53 chord
    (2, 4): {Third: 3.0, Fifth: 0.75},
    # 63 chord
    (2, 5): {Third: 2.5, Root: 1.0},
    # 64 chord
    (3, 5): {},
    # SEVENTH CHORDS
    # 73 chord
    (2, 4, 6): {Third: 2.5, Root: 1.0},
    # 65 chord
    (2, 4, 5): {Third: 3.0, Root: 1.0},
    # 43 chord
    (2, 3, 5): {Third: 2.5, Root: 1.0},
    # 42 chord
    (1, 3, 5): {Third: 2.5, Root: 0.75},
}
SUSPENSION_WEIGHTS_BY_CHROMATIC_INTERVALS: t.Mapping[
    tuple[ChromaticInterval, ...], t.Mapping
] = {
    # TRIADS
    # Diminished 63 chord
    (3, 9): {Third: 1.0, Root: 3.0},
}


def is_major_or_minor_64(chord: Chord) -> bool:
    return chord.chromatic_intervals_above_bass in ((5, 8), (5, 9))


def is_stationary_64(
    chord: Chord, previous_chord: Chord | None, next_chord: Chord | None
) -> bool:
    """
    >>> rntxt = '''m1 C: I6 b2 I b3 IV64 b4 V6/ii'''
    >>> I6, I, IV64, V6_of_ii = get_chords_from_rntxt(rntxt)
    >>> is_stationary_64(IV64, None, None)
    False
    >>> is_stationary_64(IV64, V6_of_ii, None)
    False
    >>> is_stationary_64(IV64, None, V6_of_ii)
    False
    >>> is_stationary_64(IV64, I, None)
    True
    >>> is_stationary_64(IV64, None, I)
    True
    >>> is_stationary_64(IV64, I, I)
    True
    >>> is_stationary_64(IV64, V6_of_ii, V6_of_ii)
    False
    >>> is_stationary_64(IV64, V6_of_ii, I)
    True
    >>> is_stationary_64(IV64, V6_of_ii, I6)
    False
    """
    if not is_major_or_minor_64(chord):
        return False
    if next_chord is not None:
        if next_chord.foot == chord.foot:
            return True
    if previous_chord is not None:
        if previous_chord.foot == chord.foot:
            return True
    return False


def is_63(chord: Chord, superset_ok: bool = False) -> bool:
    if superset_ok:
        return {2, 5} <= set(chord.scalar_intervals_above_bass)
    return chord.scalar_intervals_above_bass == (2, 5)


def chord_ascends_by_step(chord: Chord, next_chord: Chord) -> bool:
    return (next_chord.foot - chord.foot) % 12 in (1, 2)


def is_43_ascending_by_step(
    chord: Chord, previous_chord: Chord | None, next_chord: Chord | None
) -> bool:
    """
    >>> rntxt = "m1 C: I b2 V43 b3 I6"
    >>> I, V43, I6 = get_chords_from_rntxt(rntxt)
    >>> is_43_ascending_by_step(V43, previous_chord=I, next_chord=I6)
    True
    >>> is_43_ascending_by_step(V43, previous_chord=I6, next_chord=I)
    False
    """
    if next_chord is None:
        return False
    return chord.scalar_intervals_above_bass == (2, 3, 5) and chord_ascends_by_step(
        chord, next_chord
    )


def is_diminished_63(
    chord: Chord, previous_chord: Chord | None = None, next_chord: Chord | None = None
) -> bool:
    return chord.chromatic_intervals_above_bass == (3, 9)


def is_diminished_63_ascending_by_step_to_63(
    chord: Chord, previous_chord: Chord | None, next_chord: Chord | None
) -> bool:
    """
    >>> rntxt = '''m1 C: I b2 viio6 b3 I6 b4 vi64
    ... m2 viio6/ii b3 V43/ii'''
    >>> I, viio6, I6, vi64, viio6_of_ii, V43_of_ii = get_chords_from_rntxt(rntxt)
    >>> is_diminished_63_ascending_by_step_to_63(viio6, previous_chord=I, next_chord=I6)
    True
    >>> is_diminished_63_ascending_by_step_to_63(viio6, previous_chord=I6, next_chord=I)
    False
    >>> is_diminished_63_ascending_by_step_to_63(
    ...     viio6, previous_chord=I, next_chord=vi64
    ... )
    False
    """
    if next_chord is None:
        return False
    return (
        is_diminished_63(chord)
        and chord_ascends_by_step(chord, next_chord)
        and is_63(next_chord, superset_ok=True)
    )


CONDITIONAL_TENDENCIES: t.Mapping[
    t.Callable[[Chord, Chord | None, Chord | None], bool], AbstractChordTendencies
] = MappingProxyType(
    {
        is_stationary_64: {Root: Tendency.DOWN, Third: Tendency.DOWN},
        is_43_ascending_by_step: {
            Seventh: Tendency.UP
        },  # TODO: (Malcolm 2023-07-19) maybe change to UP
        is_diminished_63: {Fifth: Tendency.NONE},
        is_diminished_63_ascending_by_step_to_63: {Fifth: Tendency.UP},
    }
)

# TODO: (Malcolm) long run we may want to normalize these tokens somehow, so e.g.
# V643 -> V43

# VOICING_PREREQUISITES specifies chord factors that can only occur if other chord
#   factors are also present. For example, "V43": {Root: [Seventh]} specifies that
#   the root (4th above bass) can only occur in a V43 chord if the seventh (3rd above
#   bass) is also present. NB the bass is always assumed to be the prerequisite
#   for all other notes.
VOICING_PREREQUISITES: t.Mapping[
    RNToken, t.Mapping[ChordFactor, t.Sequence[ChordFactor]]
] = {"V43": {Root: [Seventh]}}


# Can't be a lambda because needs to be pickleable
def default2():
    return 2


@dataclass
class Chord:
    pcs: t.Tuple[PitchClass]
    scale_pcs: t.Tuple[PitchClass]
    onset: TimeStamp
    release: TimeStamp
    inversion: int
    token: str
    tendencies: ConcreteChordTendencies

    # whereas 'onset' and 'release' should be the start of this particular
    #   structural "unit" (which might, for example, break at a barline
    #   without a change of harmony), `harmony_onset` and `harmony_release`
    #   are for the onset and release of the harmony (i.e., the boundaries
    #   of the preceding and succeeding *changes* in the harmony)
    harmony_onset: t.Optional[TimeStamp] = field(default=None, compare=False)
    harmony_release: t.Optional[TimeStamp] = field(default=None, compare=False)

    def __post_init__(self):
        self._pc_to_bass_factor = {pc: i for (i, pc) in enumerate(self.pcs)}
        self._pc_to_chord_factor = {
            pc: (i + self.inversion) % self.cardinality
            for (i, pc) in enumerate(self.pcs)
        }
        self._pc_voicing_cache = {}
        self._max_doublings = defaultdict(default2)
        if self.is_consonant:
            # By default we permit tripling the root of consonant triads
            self._max_doublings[self.chord_factor_to_bass_factor[0]] = 3

    def __contains__(self, pitch_or_pc: PitchOrPitchClass) -> bool:
        return pitch_or_pc % 12 in self.pcs

    @classmethod
    def from_music21_rn(
        cls,
        rn: music21.roman.RomanNumeral,
        onset: TimeStamp,
        release: TimeStamp,
        token_prefix: str = "",
    ) -> Chord:
        chord_pcs = _get_chord_pcs(rn)
        scale_pcs = fit_scale_to_rn(rn)
        inversion = rn.inversion()
        tendencies = apply_tendencies(rn)
        return cls(
            pcs=chord_pcs,
            scale_pcs=scale_pcs,
            onset=onset,
            release=release,
            inversion=inversion,
            token=token_prefix + rn.figure,
            tendencies=tendencies,
        )

    @property
    def foot(self):
        return self.pcs[0]

    @property
    def non_foot_pcs(self):
        return self.pcs[1:]

    @cached_property
    def root_position_pcs(self) -> tuple[PitchClass]:
        """
        >>> rntxt = "m1 C: I b3 I6"
        >>> I, I6 = get_chords_from_rntxt(rntxt)
        >>> I.root_position_pcs
        (0, 4, 7)
        >>> I6.root_position_pcs
        (0, 4, 7)
        """
        return tuple(
            self.pcs[self.chord_factor_to_bass_factor[i]]
            for i in range(self.cardinality)
        )

    @property
    def in_root_position(self):
        return self.inversion == 0

    @property
    def scalar_intervals_including_bass(self) -> t.Tuple[ScalarInterval, ...]:
        return (0,) + self.scalar_intervals_above_bass

    @cached_property
    def scalar_intervals_above_bass(self) -> t.Tuple[ScalarInterval, ...]:
        """
        >>> rntxt = "m1 C: I b3 V43"
        >>> I, V43 = get_chords_from_rntxt(rntxt)
        >>> I.scalar_intervals_above_bass
        (2, 4)
        >>> V43.scalar_intervals_above_bass
        (2, 3, 5)
        """
        foot = self.foot
        scale_cardinality = len(self.scale_pcs)
        return tuple(
            (self.scale_pcs.index(pc) - self.scale_pcs.index(foot)) % scale_cardinality
            for pc in self.pcs[1:]
        )

    @cached_property
    def chromatic_intervals_above_bass(self) -> t.Tuple[ChromaticInterval]:
        """
        >>> rntxt = "m1 Bb: I b3 V43"
        >>> I, V43 = get_chords_from_rntxt(rntxt)
        >>> I.chromatic_intervals_above_bass
        (4, 7)
        >>> V43.chromatic_intervals_above_bass
        (3, 5, 9)
        """
        foot = self.foot
        return tuple((pc - foot) % 12 for pc in self.pcs[1:])

    def copy(self):
        """
        >>> rntxt = "m1 C: I"
        >>> (I,) = get_chords_from_rntxt(rntxt)
        >>> I.copy()  # doctest: +NORMALIZE_WHITESPACE
        Chord(pcs=(0, 4, 7), scale_pcs=(0, 2, 4, 5, 7, 9, 11), onset=Fraction(0, 1),
              release=Fraction(4, 1), inversion=0, token='C:I',
              tendencies={}, harmony_onset=Fraction(0, 1),
              harmony_release=Fraction(4, 1))
        """
        return deepcopy(self)

    @cached_property
    def _voicing_prerequisites(self) -> t.Mapping[BassFactor, t.Sequence[PitchClass]]:
        """
        >>> rntxt = "m1 C: I b3 V43"
        >>> I, V43 = get_chords_from_rntxt(rntxt)
        >>> I._voicing_prerequisites
        {}
        >>> V43._voicing_prerequisites  # pitch-class 5 (F) must be present for
        ... # bass-factor 2 (G) to be added
        {2: (5,)}
        """
        chord_factor_prerequisites = VOICING_PREREQUISITES.get(self.token, None)
        if chord_factor_prerequisites is None:
            return {}
        bass_factor_prerequisites = {
            self.chord_factor_to_bass_factor[bass_factor]: tuple(
                self.pcs[self.chord_factor_to_bass_factor[prereq]] for prereq in prereqs
            )
            for bass_factor, prereqs in chord_factor_prerequisites.items()
        }
        return bass_factor_prerequisites

    @cached_property
    def bass_factor_to_chord_factor(self) -> t.Tuple[int]:
        """
        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 V42"
        >>> V7, V65, V43, V42 = get_chords_from_rntxt(rntxt)
        >>> V7.bass_factor_to_chord_factor
        (0, 1, 2, 3)
        >>> V65.bass_factor_to_chord_factor
        (1, 2, 3, 0)
        >>> V43.bass_factor_to_chord_factor
        (2, 3, 0, 1)
        >>> V42.bass_factor_to_chord_factor
        (3, 0, 1, 2)
        """
        return tuple(
            (chord_factor_i + self.inversion) % self.cardinality
            for chord_factor_i in range(self.cardinality)
        )

    @cached_property
    def chord_factor_to_bass_factor(self) -> t.Tuple[int]:
        """
        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 V42"
        >>> V7, V65, V43, V42 = get_chords_from_rntxt(rntxt)
        >>> V7.chord_factor_to_bass_factor
        (0, 1, 2, 3)
        >>> V65.chord_factor_to_bass_factor
        (3, 0, 1, 2)
        >>> V43.chord_factor_to_bass_factor
        (2, 3, 0, 1)
        >>> V42.chord_factor_to_bass_factor
        (1, 2, 3, 0)
        """
        return tuple(
            (bass_factor_i - self.inversion) % self.cardinality
            for bass_factor_i in range(self.cardinality)
        )

    @property
    def cardinality(self):
        """
        Perhaps we could just define self.__len__, however I'm a little wary of the
        notion that a chord has a "length" since different realizations (with different
        omissions/doublings) of that chord will have different numbers of elements.

        >>> rntxt = "m1 C: V7 b3 I"
        >>> V7, I = get_chords_from_rntxt(rntxt)
        >>> V7.cardinality
        4
        >>> I.cardinality
        3
        """
        return len(self.pcs)

    @property
    def max_doublings(self):  # pylint: disable=missing-docstring
        return self._max_doublings

    def pitch_to_chord_factor(self, pitch: PitchOrPitchClass) -> ChordFactor:
        """
        >>> rntxt = "m1 C: I b3 I6"
        >>> I, I6 = get_chords_from_rntxt(rntxt)
        >>> I.pitch_to_chord_factor(60)
        0
        >>> I6.pitch_to_chord_factor(60)
        0
        """
        return self._pc_to_chord_factor[pitch % 12]

    def chord_factor_to_pc(self, chord_factor: ChordFactor) -> PitchClass:
        """
        >>> rntxt = "m1 C: I b2 I6 b3 V43"
        >>> I, I6, V43 = get_chords_from_rntxt(rntxt)
        >>> I.chord_factor_to_pc(Root)
        0
        >>> I6.chord_factor_to_pc(Root)
        0
        >>> V43.chord_factor_to_pc(Seventh)
        5

        >>> I6.chord_factor_to_pc(Seventh)
        Traceback (most recent call last):
        IndexError: tuple index out of range
        """
        return self.root_position_pcs[chord_factor]

    def pitch_to_bass_factor(self, pitch: PitchOrPitchClass) -> BassFactor:
        """
        >>> rntxt = "m1 C: I b3 I6"
        >>> I, I6 = get_chords_from_rntxt(rntxt)
        >>> I.pitch_to_bass_factor(60)
        0
        >>> I6.pitch_to_bass_factor(60)
        2
        """
        return self._pc_to_bass_factor[pitch % 12]

    def pitch_is_chord_factor(
        self, pitch: PitchOrPitchClass, chord_factor: ChordFactor
    ) -> bool:
        """
        >>> rntxt = "m1 C: I"
        >>> (I,) = get_chords_from_rntxt(rntxt)
        >>> I.pitch_is_chord_factor(60, Root)
        True
        >>> I.pitch_is_chord_factor(64, Root)
        False
        >>> I.pitch_is_chord_factor(65, Root)
        False
        """
        try:
            return self.pitch_to_chord_factor(pitch) == chord_factor
        except KeyError:
            return False

    @property
    def is_consonant(self) -> bool:
        """
        >>> rntxt = '''m1 C: I b2 IV64 b3 vi65 b4 viio7
        ... m2 I+ b2 Cad64 b3 V54 b4 V'''
        >>> I, IV64, vi65, viio7, Iaug, Cad64, V54, V = get_chords_from_rntxt(rntxt)
        >>> I.is_consonant
        True
        >>> IV64.is_consonant
        False
        >>> viio7.is_consonant
        False
        >>> Iaug.is_consonant
        False
        >>> Cad64.is_consonant
        False
        >>> V54.is_consonant
        False
        >>> V.is_consonant
        True
        """
        return pcs_consonant(self.pcs, self.scale_pcs)

    def update_tendencies_from_context(
        self, prev_chord: Chord | None, next_chord: Chord | None
    ):
        number_of_conditions_that_match = 0
        for condition, tendencies in CONDITIONAL_TENDENCIES.items():
            if condition(self, prev_chord, next_chord):
                if number_of_conditions_that_match > 0:
                    LOGGER.warning(
                        f"{number_of_conditions_that_match} conditions match chord {self}"
                    )
                tendencies = abstract_to_concrete_chord_tendencies(
                    tendencies, self.inversion, self.cardinality
                )
                self.tendencies = dict(self.tendencies) | tendencies

    def get_pcs_that_cannot_be_added_to_existing_voicing(
        self,
        existing_voices: t.Iterable[PitchOrPitchClass] = (),
        suspensions: t.Iterable[PitchOrPitchClass] = (),
    ) -> tuple[PitchClass]:
        """
        Note: `suspensions` need to be also present in
        `existing_voices_not_including_bass`.
        >>> rntxt = "m1 C: V7 b2 V65 b3 V42 b4 V43"
        >>> V7, V65, V42, V43 = get_chords_from_rntxt(rntxt)

        >>> V7.get_pcs_that_cannot_be_added_to_existing_voicing((7,))
        ()

        >>> V7.get_pcs_that_cannot_be_added_to_existing_voicing((7, 11, 2))
        (11,)

        >>> V65.get_pcs_that_cannot_be_added_to_existing_voicing((11,))
        (11,)

        >>> V42.get_pcs_that_cannot_be_added_to_existing_voicing((5, 11))
        (5, 11)

        The items in `suspensions` can be duplicated in `existing_voices` but do not
        need to be.
        >>> V42.get_pcs_that_cannot_be_added_to_existing_voicing((5,), suspensions=(0,))
        (5, 11)
        >>> V42.get_pcs_that_cannot_be_added_to_existing_voicing(
        ...     (5, 0), suspensions=(0,)
        ... )
        (5, 11)

        >>> V65.get_pcs_that_cannot_be_added_to_existing_voicing(
        ...     (48, 62), suspensions=(48,)
        ... )
        (11,)
        >>> V43.get_pcs_that_cannot_be_added_to_existing_voicing(
        ...     (51, 53), suspensions=(51,)
        ... )
        (2, 5)
        """
        if suspensions:
            existing_voices = set(existing_voices) | set(suspensions)
        omissions = self.get_omissions(existing_voices, suspensions=suspensions)
        return tuple(
            pc for pc, omission in zip(self.pcs, omissions) if omission is Allow.ONLY
        )

    def get_pcs_that_can_be_added_to_existing_voicing(
        self,
        existing_voices: t.Iterable[PitchOrPitchClass] = (),
        suspensions: t.Iterable[PitchOrPitchClass] = (),
    ) -> t.Tuple[PitchClass]:
        """
        >>> rntxt = "m1 C: V7 b2 V65 b3 V42"
        >>> V7, V65, V42 = get_chords_from_rntxt(rntxt)

        >>> V7.get_pcs_that_can_be_added_to_existing_voicing((7,))
        (7, 11, 2, 5)

        >>> V7.get_pcs_that_can_be_added_to_existing_voicing((55, 67))
        (7, 11, 2, 5)

        >>> V7.get_pcs_that_can_be_added_to_existing_voicing((7, 11, 2))
        (7, 2, 5)

        >>> V65.get_pcs_that_can_be_added_to_existing_voicing((11,))
        (2, 5, 7)

        >>> V42.get_pcs_that_can_be_added_to_existing_voicing((5, 11))
        (7, 2)

        The items in `suspensions` can be duplicated in `existing_voices` but do not
        need to be.
        >>> V42.get_pcs_that_can_be_added_to_existing_voicing((5,), suspensions=(0,))
        (7, 2)
        >>> V42.get_pcs_that_can_be_added_to_existing_voicing((5, 0), suspensions=(0,))
        (7, 2)
        """
        if suspensions:
            existing_voices = set(existing_voices) | set(suspensions)
        omissions = self.get_omissions(existing_voices, suspensions=suspensions)
        return tuple(
            pc
            for pc, omission in zip(self.pcs, omissions)
            if omission is not Allow.ONLY
        )

    def get_tendency_resolutions(
        self,
        pitch: Pitch,
        tendency: Tendency,
        resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
        # TODO: (Malcolm 2023-07-20) I formerly didn't have (2) included in upward
        # resolutions. I need to verify the results.
        resolve_up_by: t.Tuple[ChromaticInterval, ...] = (1, 2),
    ) -> Resolution | None:
        """
        >>> rntxt = '''m1 C: I7 b3 V65/ii'''
        >>> I7, V65_of_ii = get_chords_from_rntxt(rntxt)
        >>> V65_of_ii.get_tendency_resolutions(69, Tendency.NONE)

        >>> V65_of_ii.get_tendency_resolutions(68, Tendency.DOWN)
        Resolution(by=-1, to=67)
        >>> V65_of_ii.get_tendency_resolutions(68, Tendency.UP)
        Resolution(by=1, to=69)
        >>> V65_of_ii.get_tendency_resolutions(65, Tendency.UP)
        Resolution(by=2, to=67)
        >>> V65_of_ii.get_tendency_resolutions(66, Tendency.DOWN)
        Resolution(by=-2, to=64)

        In principle, at least, it's possible for a chord to contain multiple
        possible resolutions of a tendency tone. However, we only ever return one
        resolution. Resolutions are preferred in the order in which they occur in
        `resolve_down_by` or `resolve_up_by`.
        >>> I7.get_tendency_resolutions(61, Tendency.DOWN)
        ... # In theory could also resolve to `Resolution(by=-2, to=59)`
        Resolution(by=-1, to=60)
        """
        if tendency is Tendency.NONE:
            return None

        resolution_intervals = (
            resolve_down_by if tendency is Tendency.DOWN else resolve_up_by
        )

        for interval in resolution_intervals:
            if (pitch + interval) % 12 in self.pcs:
                return Resolution(by=interval, to=pitch + interval)
        return None

    def get_pitch_tendency(self, pitch: Pitch) -> Tendency:
        """
        >>> rntxt = '''m1 C: V7 b2 viio6 b3 Cad64'''
        >>> V7, viio6, Cad64 = get_chords_from_rntxt(rntxt)
        >>> V7.get_pitch_tendency(11)
        <Tendency.UP: 2>
        >>> viio6.get_pitch_tendency(5)
        <Tendency.NONE: 1>
        >>> Cad64.get_pitch_tendency(0)
        <Tendency.DOWN: 3>

        >>> rntxt = '''m1 C: I b2 IV64 b3 V6/ii b4 V64
        ... m2 I6'''
        >>> I, IV64, V6_of_ii, V64, I6 = get_chords_from_rntxt(rntxt)
        >>> IV64.get_pitch_tendency(69)
        <Tendency.DOWN: 3>
        >>> V64.get_pitch_tendency(67)
        <Tendency.NONE: 1>
        """
        bass_factor = self._pc_to_bass_factor[pitch % 12]
        return self.tendencies.get(bass_factor, Tendency.NONE)

    def pc_can_be_doubled(self, pitch_or_pc: int) -> bool:
        """
        >>> rntxt = "m1 C: V7"
        >>> (V7,) = get_chords_from_rntxt(rntxt)
        >>> V7.pc_can_be_doubled(2)
        True
        >>> V7.pc_can_be_doubled(62)
        True
        >>> V7.pc_can_be_doubled(59)
        False
        """
        return self.get_pitch_tendency(pitch_or_pc) is Tendency.NONE

    def check_pitch_doublings(self, pitches: t.Sequence[Pitch]) -> bool:
        """Checks that no tendency tones are doubled.

        >>> rntxt = "m1 C: I b2 V7 b3 viio7/vi b4 ii64"
        >>> I, V7, viio7_of_vi, ii64 = get_chords_from_rntxt(rntxt)
        >>> I.check_pitch_doublings([48, 52, 55, 60, 64, 67, 72])
        True

        Doesn't check whether chords are complete
        >>> I.check_pitch_doublings([48, 60]), I.check_pitch_doublings([])
        (True, True)

        >>> V7.check_pitch_doublings([65, 65])
        False
        >>> V7.check_pitch_doublings([55, 59, 62, 65, 67, 71])
        False
        >>> viio7_of_vi.check_pitch_doublings([56, 59, 62, 65, 71])
        True
        """
        counts = Counter([self._pc_to_bass_factor[pitch % 12] for pitch in pitches])
        for bass_factor, count in counts.items():
            if (
                count > 1
                and self.tendencies.get(bass_factor, Tendency.NONE) is not Tendency.NONE
            ):
                return False
        return True

    def pc_must_be_omitted(
        self, pc: PitchClass, existing_pitches: t.Iterable[Pitch]
    ) -> bool:
        """
        Returns true if the pc is a tendency tone that is already present
        among the existing pitches.

        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: V7
        ... m2 I'''
        >>> dom7, tonic = get_chords_from_rntxt(rntxt)
        >>> dom7.pc_must_be_omitted(11, [62, 71])  # B already present in chord
        True
        >>> dom7.pc_must_be_omitted(5, [62, 71])  # F not present in chord
        False
        >>> not any(
        ...     tonic.pc_must_be_omitted(pc, [60, 64, 67]) for pc in (0, 4, 7)
        ... )  # tonic has no tendency tones
        True

        """
        return self.get_pitch_tendency(pc) is not Tendency.NONE and any(
            pitch % 12 == pc for pitch in existing_pitches
        )

    # TODO: (Malcolm 2023-07-19) allow specifying a previous chord. Then, if
    #   the previous chords pcs are a strict subset of the current chords pcs,
    #   any additional pcs should be non-omittable (because presumably, if
    #   there is a chord change specified, whatever the notes are that indicate it
    #   should be included)
    def get_omissions(
        self,
        existing_pitches_or_pcs: t.Iterable[PitchOrPitchClass],
        suspensions: t.Iterable[PitchOrPitchClass] = (),
        iq: t.Optional[IntervalQuerier] = None,
    ) -> t.List[Allow]:
        """Get pitches that can or must be omitted based on existing pitches.

        >>> rntxt = "m1 C: V7 b2 V43 b3 I"
        >>> V7, V43, I = get_chords_from_rntxt(rntxt)

        When there are no existing pitches, all chord factors return NO:
        >>> V7.get_omissions(())
        [<Allow.NO: 1>, <Allow.NO: 1>, <Allow.NO: 1>, <Allow.NO: 1>]

        When all chord factors are among existing pitches, tendency tones are ONLY
        (they must be omitted) and other pitches are YES:
        >>> V7.get_omissions((67, 71, 74, 77))
        [<Allow.YES: 2>, <Allow.ONLY: 3>, <Allow.YES: 2>, <Allow.ONLY: 3>]

        Also works on pitch-classes:
        >>> V7.get_omissions((7, 11, 2, 5))
        [<Allow.YES: 2>, <Allow.ONLY: 3>, <Allow.YES: 2>, <Allow.ONLY: 3>]

        Return value is in "bass-factor order", i.e., in close position starting
        with the bass note.
        >>> V43.get_omissions((71,))
        [<Allow.NO: 1>, <Allow.NO: 1>, <Allow.NO: 1>, <Allow.ONLY: 3>]
        >>> V43.get_omissions((65,))
        [<Allow.NO: 1>, <Allow.ONLY: 3>, <Allow.NO: 1>, <Allow.NO: 1>]

        --------------------------------------------------------------------------------
        Case 1: no suspensions
        --------------------------------------------------------------------------------

        If there are no suspensions, then the rules are
        1. chord factors *can* be omitted (they are YES) if they satisfy
            IntervalQuerier.pc_can_be_omitted(): if there is already an imperfect
            consonance or dissonance among existing pitches OR if the pc would not
            create an imperfect consonance or a dissonance if added to the existing
            pitches.
        >>> V7.get_omissions((67,))
        [<Allow.YES: 2>, <Allow.NO: 1>, <Allow.YES: 2>, <Allow.NO: 1>]

        2. chord factors *must* be omitted (they are NO) if they are tendency tones
            found among `existing_pitches`.
        >>> V7.get_omissions((71,))
        [<Allow.NO: 1>, <Allow.ONLY: 3>, <Allow.NO: 1>, <Allow.NO: 1>]

        --------------------------------------------------------------------------------
        Case 2: suspensions
        --------------------------------------------------------------------------------

        Suspensions should be included in `existing_pitches_or_pcs` as well as in `suspensions`.
        The rules for suspensions are

        1. suspensions that resolve down by semitone to a member of the chord *must*
        be omitted:
        >>> V7.get_omissions((67, 72), suspensions=(72,))
        [<Allow.YES: 2>, <Allow.ONLY: 3>, <Allow.NO: 1>, <Allow.NO: 1>]

        2. suspensions that resolve down by wholetone to a member of the chord *can*
        be omitted:
        >>> V7.get_omissions((69, 71), suspensions=(69,))
        [<Allow.YES: 2>, <Allow.ONLY: 3>, <Allow.YES: 2>, <Allow.YES: 2>]
        """

        pitch_classes = [p % 12 for p in existing_pitches_or_pcs]
        assert all(s % 12 in pitch_classes for s in suspensions)

        if iq is None:
            iq = IntervalQuerier()

        out = []
        semitone_resolutions = {(s - 1) % 12 for s in suspensions}
        wholetone_resolutions = {(s - 2) % 12 for s in suspensions}
        for pc in self.pcs:
            if pc in semitone_resolutions or self.pc_must_be_omitted(
                pc, existing_pitches_or_pcs
            ):
                out.append(Allow.ONLY)
            elif pc in wholetone_resolutions or iq.pc_can_be_omitted(
                pc, existing_pitches_or_pcs
            ):
                out.append(Allow.YES)
            else:
                out.append(Allow.NO)
        return out

    def pitch_voicings(
        self,
        max_doubling: int | None = None,
        min_notes: int = 4,
        max_notes: int = 4,
        bass_pitch: Pitch | None = None,
        melody_pitch: Pitch | None = None,
        range_constraints: RangeConstraints = RangeConstraints(),
        spacing_constraints: SpacingConstraints = SpacingConstraints(),
        shuffled: bool = True,
    ) -> t.Iterable[t.Tuple[Pitch, ...]]:
        """
        This doesn't allow specifying suspensions, etc., because it's only intended
        to get an initial chord spacing.
        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 I"
        >>> V7, V65, V43, I = get_chords_from_rntxt(rntxt)

        >>> voicing_iter = I.pitch_voicings()
        >>> next(voicing_iter), next(voicing_iter), next(voicing_iter)  # doctest: +SKIP
        ((72, 76, 79, 79), (60, 64, 64, 67), (48, 52, 60, 64))

        >>> voicing_iter = I.pitch_voicings(bass_pitch=60, melody_pitch=76)
        >>> next(voicing_iter), next(voicing_iter), next(voicing_iter)  # doctest: +SKIP
        ((60, 64, 72, 76), (60, 60, 72, 76), (60, 60, 67, 76))

        """
        prespecified_pitches = () if melody_pitch is None else (melody_pitch,)
        pc_voicings = self.pc_voicings(
            min_notes,
            max_notes,
            max_doubling=max_doubling,
            included_factors=prespecified_pitches,
        )
        pc_voicings_weights = self.get_voicing_option_weights(
            pc_voicings,
            prespecified_pitches=prespecified_pitches,
            bass_is_included_in_voicing=True,
        )

        spacing_iters = [
            yield_spacings(
                pcs=pc_voicing,
                range_constraints=range_constraints,
                spacing_constraints=spacing_constraints,
                bass_pitch=bass_pitch,
                melody_pitch=melody_pitch,
                shuffled=shuffled,
            )
            for pc_voicing in pc_voicings
        ]
        yield from yield_sample_from_sequence_of_iters(
            spacing_iters, pc_voicings_weights
        )

    def all_pc_voicings(
        self,
        max_doubling: int | None = None,
        max_notes: int = 5,
        included_factors: t.Iterable[PitchOrPitchClass] = (),
        suspensions: t.Iterable[PitchOrPitchClass] = (),
        bass_suspension: PitchOrPitchClass | None = None,
    ) -> t.Dict[VoiceCount, t.Set[t.Tuple[PitchClass]]]:
        """
        The chords in the output are sorted in ascending pitch-class order, *except* for
        the first pc, which is always the bass. The sorting is done so that equivalent
        permuted voicings won't be considered unique.

        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 I"
        >>> V7, V65, V43, I = get_chords_from_rntxt(rntxt)
        >>> I.all_pc_voicings()  # doctest: +NORMALIZE_WHITESPACE
        {4: {(0, 0, 4, 4), (0, 0, 4, 7), (0, 0, 0, 4), (0, 4, 4, 7),
             (0, 4, 7, 7)},
         5: {(0, 0, 4, 4, 7), (0, 0, 0, 4, 7), (0, 0, 0, 4, 4),
             (0, 4, 4, 7, 7), (0, 0, 4, 7, 7)},
         3: {(0, 0, 4), (0, 4, 4), (0, 4, 7)},
         2: {(0, 4)}}


        Note: the bass is *always* an included factor. If we include a pitch with bass
        pc in `all_pc_voicings` then that pc will be included twice.
        >>> I.all_pc_voicings(max_notes=4, included_factors=(60,), max_doubling=2)
        {3: {(0, 0, 4)}, 4: {(0, 0, 4, 4), (0, 0, 4, 7)}}

        >>> V7.all_pc_voicings(
        ...     max_notes=4, max_doubling=2
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {3: {(7, 5, 11), (7, 5, 7), (7, 7, 11), (7, 2, 5), (7, 2, 11)},
         4: {(7, 2, 2, 5), (7, 2, 2, 11), (7, 2, 5, 7), (7, 2, 7, 11),
             (7, 5, 7, 11), (7, 2, 5, 11)},
         2: {(7, 5), (7, 11)}}

        >>> V7.all_pc_voicings(
        ...     max_notes=4, included_factors=(59,), max_doubling=2
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {2: {(7, 11)},
         3: {(7, 7, 11), (7, 5, 11), (7, 2, 11)},
         4: {(7, 5, 7, 11), (7, 2, 2, 11), (7, 2, 7, 11), (7, 2, 5, 11)}}

        >>> V7.all_pc_voicings(
        ...     max_notes=4, suspensions=(60,), max_doubling=2
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {4: {(7, 0, 5, 7), (7, 0, 2, 5), (7, 0, 2, 7), (7, 0, 2, 2)},
         3: {(7, 0, 2), (7, 0, 5)}}

        A non-bass suspension that is not a semi-tone above any chord factor has no
        effect on the inclusion/doubling of the chord factors.
        >>> I.all_pc_voicings(
        ...     max_notes=4, suspensions=(71,), max_doubling=2
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {2: {(0, 11)},
         3: {(0, 0, 11), (0, 4, 11), (0, 7, 11)},
         4: {(0, 0, 7, 11), (0, 4, 4, 11), (0, 4, 7, 11), (0, 0, 4, 11), (0, 7, 7, 11)}}

        Note: this means that whole-tones below suspensions can and will be included.
        But in fact we only want to admit whole-tones below suspensions under certain
        circumstances (i.e., that they occur in a different octave). This function
        doesn't seem to be the appropriate place to test that since it deals with pitch-
        classes only.

        Bass suspensions:
        >>> I.all_pc_voicings(
        ...     max_notes=4, bass_suspension=50
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {2: {(2, 4), (2, 0)},
         3: {(2, 4, 7), (2, 4, 4), (2, 0, 7), (2, 0, 4), (2, 0, 0)},
         4: {(2, 0, 0, 4), (2, 0, 0, 7), (2, 4, 7, 7), (2, 0, 7, 7),
             (2, 0, 4, 4), (2, 0, 4, 7), (2, 4, 4, 7)}}

        --------------------------------------------------------------------------------
        Voicing prerequisites
        --------------------------------------------------------------------------------

        Voicing "prerequisites" can be specified by editing VOICING_PREREQUISITES.
        These indicate chord-factors that can only be included after other chord-factors
        are already present. For example, In V43, we only include the root if
        the seventh is present (note there are no voicings w/ pitch class 7 but
        omitting pitch class 5):
        >>> V43.all_pc_voicings(
        ...     max_notes=4, suspensions=[60]
        ... )  # doctest: +NORMALIZE_WHITESPACE
        {2: {(2, 0)},
         3: {(2, 0, 2), (2, 0, 5)},
         4: {(2, 0, 2, 5), (2, 0, 5, 7)}}

        """
        args = (
            max_doubling,
            max_notes,
            tuple(included_factors),
            tuple(suspensions),
            bass_suspension,
        )

        if args in self._pc_voicing_cache:
            return self._pc_voicing_cache[args].copy()

        working_area: t.DefaultDict[
            VoiceCount, t.Set[t.Tuple[PitchClass]]
        ] = defaultdict(set)
        voicing = (
            [self.pcs[0] if bass_suspension is None else bass_suspension % 12]
            + [s % 12 for s in suspensions]
            + [p % 12 for p in included_factors]
        )

        counts = Counter([0] + [self.pitch_to_bass_factor(p) for p in included_factors])

        all_suspensions = tuple(suspensions) + (
            (bass_suspension,) if bass_suspension is not None else ()
        )
        voicing_prequisites = self._voicing_prerequisites

        def _recurse(voicing: list):
            if (
                len(voicing) in working_area
                and tuple(voicing) in working_area[len(voicing)]
            ):
                return
            omissions = self.get_omissions(voicing, suspensions=all_suspensions)
            if not any(omission is Allow.NO for omission in omissions):
                working_area[len(voicing)].add(tuple(voicing))
            if len(voicing) >= max_notes:
                return
            for bass_factor_i, omission in enumerate(omissions):
                if bass_factor_i in voicing_prequisites:
                    # TODO: (Malcolm) perhaps any is better, so prerequisites are
                    #   sufficient rather than necessary?
                    if not all(
                        pitch_class_among_pitches(pc, voicing)
                        for pc in voicing_prequisites[bass_factor_i]
                    ):
                        continue
                if counts[bass_factor_i] >= (
                    max_doubling
                    if max_doubling is not None
                    else self.max_doublings[bass_factor_i]
                ):
                    continue
                if omission is not Allow.ONLY:
                    counts[bass_factor_i] += 1
                    _recurse(
                        voicing[:1] + sorted(voicing[1:] + [self.pcs[bass_factor_i]])
                    )
                    counts[bass_factor_i] -= 1

        _recurse(voicing)

        # I was casting to MappingProxyType here but that caused Chord.copy() to fail
        # because MappingProxyType can't be pickled. Instead we always copy the dict
        # before returning it so that modifications won't modify the cache. Note that
        # the keys and values are both immutable so we don't need to worry about them
        # being modified.
        out = dict(working_area)
        self._pc_voicing_cache[args] = out
        return out.copy()

    def pc_voicings(
        self,
        min_notes: int,
        max_notes: int,
        max_doubling: int | None = None,
        included_factors: t.Iterable[PitchOrPitchClass] = (),
        suspensions: t.Iterable[PitchOrPitchClass] = (),
    ) -> t.Set[t.Tuple[PitchClass]]:
        """Convenience function to get a set of voicings for a range of numbers of notes.

        >>> rntxt = "m1 C: V7 b3 I"
        >>> V7, I = get_chords_from_rntxt(rntxt)

        >>> I.pc_voicings(min_notes=2, max_notes=3)  # doctest: +SKIP
        {(0, 0, 4), (0, 4, 4), (0, 4), (0, 4, 7)}
        """
        voicings = self.all_pc_voicings(
            max_doubling=max_doubling,
            max_notes=max_notes,
            included_factors=included_factors,
            suspensions=suspensions,
        )
        out = set()
        for num_notes in range(min_notes, max_notes + 1):
            if num_notes in voicings:
                out |= voicings[num_notes]
        return out

    def get_pcs_needed_to_complete_voicing(
        self,
        other_chord_factors: t.Iterable[Pitch] = (),
        suspensions: t.Iterable[Pitch] = (),
        bass_suspension: PitchOrPitchClass | None = None,
        min_notes: int = 1,
        max_notes: int = 4,
        max_doubling: int | None = None,
    ) -> t.List[t.List[PitchClass]]:
        """

        Note: the bass is not included in the result.

        # TODO: (Malcolm 2023-07-23) restore
        # >>> rntxt = "m1 F: viio6/ii"
        # >>> (viio6_of_ii,) = get_chords_from_rntxt(rntxt)
        # >>> viio6_of_ii.get_pcs_needed_to_complete_voicing(
        # ...     other_chord_factors=(66,), bass_suspension=46, min_notes=4, max_notes=4
        # ... )

        >>> rntxt = '''m1 C: V7 b2 V65 b3 V6 b4 V43
        ... m2 I'''
        >>> V7, V65, V6, V43, I = get_chords_from_rntxt(rntxt)

        >>> V7.get_pcs_needed_to_complete_voicing()  # doctest: +NORMALIZE_WHITESPACE
        [[5], [11], [5, 11], [5, 7], [7, 11], [2, 5], [2, 11], [2, 2, 5], [2, 2, 11],
         [2, 5, 7], [2, 7, 11], [5, 7, 11], [2, 5, 11]]
        >>> I.get_pcs_needed_to_complete_voicing(
        ...     other_chord_factors=(67,), max_doubling=1
        ... )
        [[4]]

        Suspension:
        >>> V43.get_pcs_needed_to_complete_voicing(
        ...     suspensions=[60], min_notes=4, max_notes=4
        ... )
        [[2, 5], [5, 7]]

        >>> V7.get_pcs_needed_to_complete_voicing(suspensions=[60])
        [[2], [5], [5, 7], [2, 5], [2, 7], [2, 2]]


        Bass suspension:
        >>> V65.get_pcs_needed_to_complete_voicing(
        ...     bass_suspension=60
        ... )  # doctest: +NORMALIZE_WHITESPACE
        [[2], [2, 2], [5, 7], [2, 7], [2, 5], [5, 7, 7], [2, 5, 7], [2, 2, 5],
         [2, 7, 7], [2, 2, 7]]
        """

        all_existing = (
            ([self.pcs[0]] if bass_suspension is None else [bass_suspension])
            + list(other_chord_factors)
            + list(suspensions)
        )
        all_pc_voicings = self.all_pc_voicings(
            max_doubling=max_doubling,
            max_notes=max_notes,
            included_factors=other_chord_factors,
            suspensions=suspensions,
            bass_suspension=bass_suspension,
        )
        out = []
        for num_notes in range(min_notes, max_notes + 1):
            if num_notes not in all_pc_voicings:
                continue
            for voicing in all_pc_voicings[num_notes]:
                try:
                    pc_complement = get_pc_complement(
                        voicing, all_existing, raise_exception=True
                    )
                except ValueError:
                    pass
                else:
                    out.append(pc_complement)
        return out

    def get_voicing_option_weights(
        self,
        voicing_options: t.Iterable[t.Iterable[PitchOrPitchClass]],
        prespecified_pitches: t.Iterable[PitchOrPitchClass] = (),
        prefer_to_omit_pcs: t.Iterable[PitchClass] = (),
        bass_is_included_in_voicing: bool = False,
        max_score: float = 2.0,
    ) -> t.List[float]:
        """
        The score is calculated as follows:
        1. find the maximum number of distinct pcs among the voicings (3 in the following
            example, because the bass is pc 4), excluding any pcs in `prefer_to_omit_pcs`.
        2. this number of pcs is given `max_score` (by default 2.0).
        3. For each option, we
            a. subtract the difference between its pc count and the maximum found at 1. above
            b. subtract the number of (distinct) pcs in `prefer_to_omit_pcs`
        >>> rntxt = '''m1 C: I6 b2 IV b3 V7'''
        >>> I6, IV, V7 = get_chords_from_rntxt(rntxt)
        >>> I6.get_voicing_option_weights([[7, 0, 0], [0, 0, 0]])
        [2.0, 1.0]

        This procedure allows us to calculate sensible scores even when some pitches *must*
        be missing due to suspensions, etc. For example, pitch-class 0 does not occur in any
        of the voicings in the next example. This would be likely to occur if there was a
        D-C suspension in another voice.
        >>> I6.get_voicing_option_weights([[7, 4], [7, 7], [4, 4]])
        [2.0, 2.0, 1.0]

        >>> V7.get_voicing_option_weights(
        ...     [[7, 11, 2], [11, 2, 5], [11, 7, 7], [7, 7, 7]]
        ... )
        [1.0, 2.0, 0.0, -1.0]

        >>> IV.get_voicing_option_weights(
        ...     [[5, 9, 0], [5, 0, 0], [5, 5, 0]], prefer_to_omit_pcs=(9,)
        ... )
        [1.0, 2.0, 2.0]
        """
        if not voicing_options:
            return []
        pc_counts = []
        penalties = []

        for voicing_option in voicing_options:
            pc_count = 0.0
            penalty = 0.0
            for pc in set(
                p % 12
                for p in chain(
                    voicing_option,
                    prespecified_pitches,
                    () if bass_is_included_in_voicing else (self.foot,),
                )
            ):
                if pc in prefer_to_omit_pcs:
                    penalty += 1.0
                else:
                    pc_count += 1.0
            pc_counts.append(pc_count)
            penalties.append(penalty)

        diffs: list[int] = [self.cardinality - pc_count for pc_count in pc_counts]
        min_diff = min(diffs)
        out = [
            float(max_score + min_diff - diff - penalty)
            for diff, penalty in zip(diffs, penalties)
        ]
        return out

    @cached_property
    def suspension_weight_per_chord_factor(self) -> defaultdict[ChordFactor, float]:
        out_dict = {}
        if self.scalar_intervals_above_bass in SUSPENSION_WEIGHTS_BY_SCALAR_INTERVALS:
            out_dict |= SUSPENSION_WEIGHTS_BY_SCALAR_INTERVALS[
                self.scalar_intervals_above_bass
            ]
        if (
            self.chromatic_intervals_above_bass
            in SUSPENSION_WEIGHTS_BY_CHROMATIC_INTERVALS
        ):
            out_dict |= SUSPENSION_WEIGHTS_BY_CHROMATIC_INTERVALS[
                self.chromatic_intervals_above_bass
            ]
        return defaultdict(lambda: DEFAULT_CHORD_FACTOR_SUSPENSION_WEIGHT, out_dict)

    @cached_property
    def augmented_second_adjustments(self) -> t.Dict[ScaleDegree, Inflection]:
        """
        Suppose that a scale contains an augmented second. Then, to remove the
        augmented 2nd
            - if both notes are members of the chord, we should not remove
                the augmented 2nd (musically speaking this isn't necessarily
                so but it seems like a workable assumption for now)
            - if the higher note is a member of the chord, we raise the lower
                note
            - if the lower note is a member of the chord, we lower the higher
                note
            - if neither note is a member of the chord, we can adjust either
                note, depending on the direction of melodic motion

        This function assumes that there are no consecutive augmented seconds
        in the scale.

        >>> rntxt = '''Time Signature: 4/4
        ... m1 a: V7
        ... m2 viio7
        ... m3 i
        ... m4 iv'''
        >>> dom7, viio7, i, iv = get_chords_from_rntxt(rntxt)
        >>> dom7.augmented_second_adjustments  # ^6 should be raised
        {5: <Inflection.UP: 3>, 6: <Inflection.NONE: 2>}
        >>> viio7.augmented_second_adjustments  # both ^6 and ^7 are chord tones
        {5: <Inflection.NONE: 2>, 6: <Inflection.NONE: 2>}
        >>> i.augmented_second_adjustments  # no augmented 2nd
        {}
        >>> i.scale_pcs = (9, 11, 0, 2, 4, 5, 8)  # harmonic-minor scale
        >>> del i.augmented_second_adjustments  # rebuild augmented_second_adjustments
        >>> i.augmented_second_adjustments
        {5: <Inflection.EITHER: 1>, 6: <Inflection.EITHER: 1>}
        >>> iv.scale_pcs = (9, 11, 0, 2, 4, 5, 8)  # harmonic-minor scale
        >>> iv.augmented_second_adjustments
        {5: <Inflection.NONE: 2>, 6: <Inflection.DOWN: 4>}
        """
        out = {}
        for i, (pc1, pc2) in enumerate(
            zip(self.scale_pcs, self.scale_pcs[1:] + (self.scale_pcs[0],))
        ):
            if (pc2 - pc1) % 12 > 2:
                if pc2 in self.pcs:
                    if pc1 in self.pcs:
                        out[i] = Inflection.NONE
                    else:
                        out[i] = Inflection.UP
                    out[(i + 1) % len(self.scale_pcs)] = Inflection.NONE
                elif pc1 in self.pcs:
                    out[i] = Inflection.NONE
                    out[(i + 1) % len(self.scale_pcs)] = Inflection.DOWN
                else:
                    out[i] = Inflection.EITHER
                    out[(i + 1) % len(self.scale_pcs)] = Inflection.EITHER
        return out

    def transpose(self, interval: ChromaticInterval) -> Chord:
        """
        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: I
        ... m2 V6
        ... m3 G: V65'''
        >>> chord1, chord2, chord3 = get_chords_from_rntxt(rntxt)
        >>> chord1.transpose(3).token
        'Eb:I'
        >>> chord2.transpose(3).token
        'V6'
        >>> chord2.transpose(3).pcs
        (2, 5, 10)
        >>> chord3.transpose(4).token
        'B:V65'
        """
        out = copy.copy(self)
        out.pcs = tuple((pc + interval) % 12 for pc in out.pcs)
        out.scale_pcs = tuple((pc + interval) % 12 for pc in out.scale_pcs)
        if ":" in out.token:
            # we need to transpose the key symbol
            m = re.match(r"(?P<key>[^:]+):(?P<numeral>.*)", out.token)
            assert m is not None
            key = speller_pcs(
                (unspeller_pcs(m.group("key")) + interval) % 12  # type:ignore
            )
            out.token = key + ":" + m.group("numeral")  # type:ignore
        out._pc_to_bass_factor = {pc: i for (i, pc) in enumerate(out.pcs)}
        return out

    def scalar_intervals_from_bass_factor_to_others(
        self, bass_factor: BassFactor
    ) -> tuple[ScalarInterval, ...]:
        """
        >>> rntxt = "m1 C: I b2 I6 b3 I64 b4 V43"
        >>> I, I6, I64, V43 = get_chords_from_rntxt(rntxt)
        >>> I.scalar_intervals_from_bass_factor_to_others(0)
        (-5, -3, 2, 4)
        >>> I.scalar_intervals_from_bass_factor_to_others(1)
        (-5, -2, 2, 5)
        >>> I.scalar_intervals_from_bass_factor_to_others(2)
        (-4, -2, 3, 5)
        """
        scale_card = len(self.scale_pcs)
        scalar_intervals = self.scalar_intervals_including_bass
        bass_factor_as_scalar_interval = scalar_intervals[bass_factor]
        up = tuple(
            sorted(
                (scalar_interval - bass_factor_as_scalar_interval) % scale_card
                for scalar_interval in scalar_intervals
                if scalar_interval != bass_factor_as_scalar_interval
            )
        )
        down = tuple(f - scale_card for f in up)

        return down + up


def is_same_harmony(
    chord1: Chord,
    chord2: Chord,
    compare_scales: bool = True,
    compare_inversions: bool = True,
    allow_subsets: bool = False,
) -> bool:
    """
    >>> rntxt = '''Time Signature: 4/4
    ... m1 C: I
    ... m2 I b3 I6
    ... m3 V7/IV
    ... m4 F: V'''
    >>> I, Ib, I6, V7_of_IV, F_V = get_chords_from_rntxt(rntxt)
    >>> is_same_harmony(I, Ib)
    True
    >>> is_same_harmony(I, I6)
    False
    >>> is_same_harmony(I, I6, compare_inversions=False)
    True
    >>> is_same_harmony(I, V7_of_IV, allow_subsets=True)
    False
    >>> is_same_harmony(I, V7_of_IV, compare_scales=False, allow_subsets=True)
    True
    >>> is_same_harmony(I, F_V, compare_scales=False)
    True
    >>> is_same_harmony(I, F_V, compare_scales=True)
    False
    >>> is_same_harmony(V7_of_IV, F_V, allow_subsets=True)
    True
    """
    if compare_inversions:
        if allow_subsets:
            if chord1.pcs[0] != chord2.pcs[0] or len(
                set(chord1.pcs) | set(chord2.pcs)
            ) > max(len(chord1.pcs), len(chord2.pcs)):
                return False
            if compare_scales:
                if chord1.scale_pcs[0] != chord2.scale_pcs[0] or len(
                    set(chord1.scale_pcs) | set(chord2.scale_pcs)
                ) > max(len(chord1.scale_pcs), len(chord2.scale_pcs)):
                    return False
        else:
            if chord1.pcs != chord2.pcs:
                return False
            if compare_scales:
                if chord1.scale_pcs != chord2.scale_pcs:
                    return False
    else:
        if allow_subsets:
            if len(set(chord1.pcs) | set(chord2.pcs)) > max(
                len(chord1.pcs), len(chord2.pcs)
            ):
                return False
            if compare_scales:
                if len(set(chord1.scale_pcs) | set(chord2.scale_pcs)) > max(
                    len(chord1.scale_pcs), len(chord2.scale_pcs)
                ):
                    return False
        else:
            if set(chord1.pcs) != set(chord2.pcs):
                return False
            if compare_scales:
                if set(chord1.scale_pcs) != set(chord2.scale_pcs):
                    return False
    return True


def get_rn_without_figure(rn: music21.roman.RomanNumeral):
    """It seems that music21 doesn't provide a method for returning everything
    *but* the numeric figures from a roman numeral token.

    >>> RN = music21.roman.RomanNumeral
    >>> get_rn_without_figure(RN("V6"))
    'V'
    >>> get_rn_without_figure(RN("V+6"))
    'V+'
    >>> get_rn_without_figure(RN("viio6"))
    'viio'
    >>> get_rn_without_figure(RN("Cad64"))
    'Cad'
    """
    if rn.figure.startswith("Cad"):
        # Cadential 6/4 chord is a special case: rn.primaryFigure will return "I"
        return "Cad"
    return rn.primaryFigure.rstrip("0123456789/")


def abstract_to_concrete_chord_tendencies(
    abstract_chord_tendencies: AbstractChordTendencies, inversion: int, cardinality: int
) -> ConcreteChordTendencies:
    return {
        (chord_factor_i - inversion)
        % cardinality: abstract_chord_tendencies[chord_factor_i]
        for chord_factor_i in range(cardinality)
        if chord_factor_i in abstract_chord_tendencies
    }


def apply_tendencies(
    rn: music21.roman.RomanNumeral,
    tendencies: t.Mapping[RNTokenWithoutFigure, AbstractChordTendencies] = TENDENCIES,
) -> ConcreteChordTendencies:
    """
    Keys of returned dict are BassFactors (i.e., the pcs in
    close position with the bass as the first element).

    >>> RN = music21.roman.RomanNumeral
    >>> apply_tendencies(RN("V"))
    {1: <Tendency.UP: 2>}
    >>> apply_tendencies(RN("V7"))
    {1: <Tendency.UP: 2>, 3: <Tendency.DOWN: 3>}
    >>> apply_tendencies(RN("V42"))
    {2: <Tendency.UP: 2>, 0: <Tendency.DOWN: 3>}
    >>> apply_tendencies(RN("I"))
    {}
    >>> apply_tendencies(RN("Ger65"))
    {3: <Tendency.UP: 2>, 0: <Tendency.DOWN: 3>}
    >>> apply_tendencies(RN("Fr43"))
    {3: <Tendency.UP: 2>, 0: <Tendency.DOWN: 3>}
    >>> apply_tendencies(RN("viio6"))
    {2: <Tendency.UP: 2>, 1: <Tendency.DOWN: 3>}
    >>> apply_tendencies(RN("Cad64"))
    {1: <Tendency.DOWN: 3>, 2: <Tendency.DOWN: 3>}
    """
    inversion = rn.inversion()
    cardinality = rn.pitchClassCardinality
    figure = get_rn_without_figure(rn)
    if figure not in tendencies:
        return {}
    raw_tendencies = tendencies[figure]
    return abstract_to_concrete_chord_tendencies(raw_tendencies, inversion, cardinality)


def fit_scale_to_rn(rn: music21.roman.RomanNumeral) -> t.Tuple[PitchClass]:
    """
    >>> RN = music21.roman.RomanNumeral
    >>> fit_scale_to_rn(RN("viio7", keyOrScale="C"))  # Note A-flat
    (0, 2, 4, 5, 7, 8, 11)
    >>> fit_scale_to_rn(RN("Ger6", keyOrScale="C"))  # Note A-flat, E-flat, F-sharp
    (0, 2, 3, 6, 7, 8, 11)


    If the roman numeral has a secondary key, we use that as the scale.
    TODO I'm not sure this is always desirable.

    >>> fit_scale_to_rn(RN("V/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 4, 6)
    >>> fit_scale_to_rn(RN("viio7/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 3, 6)
    >>> fit_scale_to_rn(RN("viio7/bIII", keyOrScale="C"))  # Note C-flat
    (3, 5, 7, 8, 10, 11, 2)

    Sometimes flats are indicated for chord factors that are already flatted
    in the relevant scale. We handle those with a bit of a hack:
    >>> fit_scale_to_rn(RN("Vb9", keyOrScale="c"))
    (0, 2, 3, 5, 7, 8, 11)
    >>> fit_scale_to_rn(RN("Vb9/vi", keyOrScale="C"))
    (9, 11, 0, 2, 4, 5, 8)

    There can be a similar issue with raised degrees. If the would-be raised
    degree is already in the scale, we leave it unaltered:
    >>> fit_scale_to_rn(RN("V+", keyOrScale="c"))
    (0, 2, 3, 5, 7, 8, 11)
    """

    def _add_inflection(degree: ScaleDegree, inflection: ChromaticInterval):
        inflected_pitch = (scale_pcs[degree] + inflection) % 12
        if (
            inflection < 0
            and inflected_pitch == scale_pcs[(degree - 1) % len(scale_pcs)]
        ):
            # hack to handle "b9" etc. when already in scale
            return
        if (
            inflection > 0
            and inflected_pitch == scale_pcs[(degree + 1) % len(scale_pcs)]
        ):
            return
        scale_pcs[degree] = inflected_pitch

    if rn.secondaryRomanNumeralKey is None:
        key = rn.key
    else:
        key = rn.secondaryRomanNumeralKey
    # music21 returns the scale *including the upper octave*, which we do
    #   not want
    scale_pcs = [p.pitchClass for p in key.pitches[:-1]]  # type:ignore
    for pitch in rn.pitches:
        # NB degrees are 1-indexed so we must subtract 1 below
        degree, accidental = key.getScaleDegreeAndAccidentalFromPitch(  # type:ignore
            pitch
        )
        if accidental is not None:
            inflection = int(accidental.alter)
            _add_inflection(degree - 1, inflection)  # type:ignore
    try:
        assert len(scale_pcs) == len(set(scale_pcs))
    except AssertionError:
        # these special cases are hacks until music21's RomanNumeral
        #   handling is repaired or I figure out another solution
        if rn.figure == "bII7":
            scale_pcs = [p.pitchClass for p in key.pitches[:-1]]  # type:ignore
            _add_inflection(1, -1)
            _add_inflection(5, -1)
        else:
            raise
    return tuple(scale_pcs)


def _get_chord_pcs(rn: music21.roman.RomanNumeral) -> t.Tuple[PitchClass]:
    # remove this function after music21's RomanNumeral
    #   handling is repaired or I figure out another solution
    def _transpose(pcs, tonic_pc):
        return tuple((pc + tonic_pc) % 12 for pc in pcs)

    if rn.figure == "bII7":
        return _transpose((1, 5, 8, 0), rn.key.tonic.pitchClass)  # type:ignore
    else:
        return tuple(rn.pitchClasses)


def get_harmony_onsets_and_releases(chord_list: t.List[Chord]):
    def _clear_accumulator():
        nonlocal accumulator, prev_chord, release, onset
        for accumulated in accumulator:
            accumulated.harmony_onset = onset
            accumulated.harmony_release = release
        accumulator = []
        prev_chord = chord

    prev_chord = None
    onset = None
    release = None
    accumulator = []
    for chord in chord_list:
        if chord != prev_chord:
            if release is not None:
                _clear_accumulator()
            onset = chord.onset
        accumulator.append(chord)
        release = chord.release
    _clear_accumulator()


def strip_added_tones(rn_data: str) -> str:
    """
    >>> rntxt = '''m1 f: i b2 V7[no3][add4] b2.25 V7[no5][no3][add6][add4]
    ... m2 Cad64 b1.75 V b2 i[no3][add#7][add4] b2.5 i[add9] b2.75 i'''
    >>> print(strip_added_tones(rntxt))
    m1 f: i b2 V7 b2.25 V7
    m2 Cad64 b1.75 V b2 i b2.5 i b2.75 i
    """

    if os.path.exists(rn_data):
        with open(rn_data) as inf:
            rn_data = inf.read()
    return re.sub(r"\[(no|add)[^\]]+\]", "", rn_data)


@cacher()
def get_chords_from_rntxt(
    rn_data: str,
    split_chords_at_metric_strong_points: bool = True,
    no_added_tones: bool = True,
) -> t.List[Chord]:
    if no_added_tones:
        rn_data = strip_added_tones(rn_data)
    score = parse_rntxt(rn_data)
    m21_ts = score[music21.meter.TimeSignature].first()  # type:ignore
    ts = f"{m21_ts.numerator}/{m21_ts.denominator}"  # type:ignore
    ts = Meter(ts)
    duration = start = pickup_offset = key = None
    prev_chord = None
    out_list = []
    for rn in score.flatten()[music21.roman.RomanNumeral]:
        if pickup_offset is None:
            pickup_offset = TIME_TYPE(((rn.beat) - 1) * rn.beatDuration.quarterLength)
        start = TIME_TYPE(rn.offset) + pickup_offset
        duration = TIME_TYPE(rn.duration.quarterLength)
        if rn.key.tonicPitchNameWithCase != key:  # type:ignore
            key = rn.key.tonicPitchNameWithCase  # type:ignore
            token_prefix = key + ":"
        else:
            token_prefix = ""
        chord = Chord.from_music21_rn(
            rn, onset=start, release=start + duration, token_prefix=token_prefix
        )
        if prev_chord is not None and is_same_harmony(prev_chord, chord):
            prev_chord.release += TIME_TYPE(rn.duration.quarterLength)
        else:
            if prev_chord is not None:
                out_list.append(prev_chord)
            prev_chord = chord

    if prev_chord is not None:
        out_list.append(prev_chord)

    if split_chords_at_metric_strong_points:
        chord_list = ts.split_at_metric_strong_points(
            out_list, min_split_dur=ts.beat_dur
        )
        out_list = []
        for chord_pcs in chord_list:
            out_list.extend(ts.split_odd_duration(chord_pcs, min_split_dur=ts.beat_dur))

    get_harmony_onsets_and_releases(out_list)

    for prev_chord, chord, next_chord in zip(
        [None] + out_list[:-1], out_list, out_list[1:] + [None]
    ):
        chord.update_tendencies_from_context(prev_chord, next_chord)

    return out_list


def ascending_chord_intervals(
    intervals: t.Sequence[ScalarInterval], n_steps_per_octave: int = 7
) -> t.Iterator[ScalarInterval]:
    """
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([0, 2, 4]))]
    [0, 2, 4, 7, 9, 11, 14, 16]
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([0, 1, 3, 5]))]
    [0, 1, 3, 5, 7, 8, 10, 12]
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([1, 2, 4]))]
    [1, 2, 4, 8, 9, 11, 15, 16]
    """
    card = len(intervals)
    for interval, octave in zip(
        cycle(intervals), chain.from_iterable(repeat(o, card) for o in count())
    ):
        yield interval + octave * n_steps_per_octave


def ascending_chord_intervals_within_range(
    intervals: t.Sequence[ScalarInterval],
    inclusive_endpoint: ScalarInterval,
    n_steps_per_octave: int = 7,
):
    """
    >>> list(ascending_chord_intervals_within_range([0, 2, 4], 0))
    [0]
    >>> list(ascending_chord_intervals_within_range([0, 2, 4], 15))
    [0, 2, 4, 7, 9, 11, 14]
    """
    for x in ascending_chord_intervals(intervals, n_steps_per_octave):
        if x > inclusive_endpoint:
            return
        yield x


def descending_chord_intervals(
    intervals: t.Sequence[ScalarInterval], n_steps_per_octave: int = 7
) -> t.Iterator[ScalarInterval]:
    """
    >>> [i for _, i in zip(range(8), descending_chord_intervals([0, 2, 4]))]
    [0, -3, -5, -7, -10, -12, -14, -17]
    >>> [i for _, i in zip(range(8), descending_chord_intervals([0, 1, 3, 5]))]
    [0, -2, -4, -6, -7, -9, -11, -13]
    >>> [i for _, i in zip(range(8), descending_chord_intervals([1, 2, 4]))]
    [-3, -5, -6, -10, -12, -13, -17, -19]
    """
    if intervals[0] == 0:
        yield 0

    card = len(intervals)
    for interval, octave in zip(
        cycle(reversed(intervals)),
        chain.from_iterable(repeat(o, card) for o in count(start=1)),
    ):
        yield interval - octave * n_steps_per_octave


def descending_chord_intervals_within_range(
    intervals: t.Sequence[ScalarInterval],
    inclusive_endpoint: ScalarInterval,
    n_steps_per_octave: int = 7,
):
    """
    >>> list(descending_chord_intervals_within_range([0, 2, 4], 0))
    [0]
    >>> list(descending_chord_intervals_within_range([0, 2, 4], -15))
    [0, -3, -5, -7, -10, -12, -14]
    """
    for x in descending_chord_intervals(intervals, n_steps_per_octave):
        if x < inclusive_endpoint:
            return
        yield x


# doctests in cached_property methods are not discovered and need to be
#   added explicitly to __test__; see https://stackoverflow.com/a/72500890/10155119
__test__ = {
    "Chord.root_position_pcs": Chord.root_position_pcs,
    "Chord.scalar_intervals_above_bass": Chord.scalar_intervals_above_bass,
    "Chord.chromatic_intervals_above_bass": Chord.chromatic_intervals_above_bass,
    "Chord.augmented_second_adjustments": Chord.augmented_second_adjustments,
    "Chord.bass_factor_to_chord_factor": Chord.bass_factor_to_chord_factor,
    "Chord.chord_factor_to_bass_factor": Chord.chord_factor_to_bass_factor,
    "Chord._voicing_prerequisites": Chord._voicing_prerequisites,
}
