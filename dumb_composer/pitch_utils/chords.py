from __future__ import annotations

import copy
import os
import re
import typing as t
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from functools import cached_property
from numbers import Number
from types import MappingProxyType

import music21
from cache_lib import cacher
from voice_leader import voice_lead_pitches_multiple_options_iter

from dumb_composer.constants import TIME_TYPE, speller_pcs, unspeller_pcs
from dumb_composer.pitch_utils.aliases import Fifth, Root, Seventh, Third
from dumb_composer.pitch_utils.consonances import pcs_consonant
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.music21_handler import parse_rntxt
from dumb_composer.pitch_utils.parts import (
    outer_voices_have_forbidden_antiparallels,
    succession_has_forbidden_parallels,
)
from dumb_composer.pitch_utils.pcs import get_pc_complement, pitch_class_among_pitches
from dumb_composer.pitch_utils.spacings import (
    RangeConstraints,
    SpacingConstraints,
    validate_spacing,
    yield_spacings,
)
from dumb_composer.pitch_utils.types import (
    BassFactor,
    ChordFactor,
    ChromaticInterval,
    Pitch,
    PitchClass,
    PitchOrPitchClass,
    RNToken,
    ScaleDegree,
    TimeStamp,
    VoiceCount,
)
from dumb_composer.suspensions import Suspension
from dumb_composer.time import Meter
from dumb_composer.utils.iterables import yield_from_sequence_of_iters


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


class Inflection(Enum):
    EITHER = auto()
    NONE = auto()
    UP = auto()
    DOWN = auto()


@dataclass
class Resolution:
    by: ChromaticInterval
    to: PitchClass


# TODO: (Malcolm) allow to override tendencies for specific inversions, bass motions,
# etc. For example, in V43 going to I6, the seventh (3rd above bass) of the V should
# not typically descend.
# Likewise, it is not only cadential 64 chords where we want to have the specified
# tendencies, but also neighbor 64 chords.
TENDENCIES = MappingProxyType(
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
        # To get the correct tendencies for the cadential 64 chord we need
        #   to index into it as I. Sorry Schenkerians!
        "Cad": {Root: Tendency.DOWN, Third: Tendency.DOWN},
        # TODO: (Malcolm 2023-07-12) test N
        "N": {Root: Tendency.DOWN},
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
] = MappingProxyType({"V43": {Root: [Seventh]}})


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
    intervals_above_bass: t.Tuple[int]
    tendencies: t.Dict[ChordFactor, Tendency]

    # whereas 'onset' and 'release' should be the start of this particular
    #   structural "unit" (which might, for example, break at a barline
    #   without a change of harmony), `harmony_onset` and `harmony_release`
    #   are for the onset and release of the harmony (i.e., the boundaries
    #   of the preceding and succeeding *changes* in the harmony)
    harmony_onset: t.Optional[TimeStamp] = field(default=None, compare=False)
    harmony_release: t.Optional[TimeStamp] = field(default=None, compare=False)

    def __post_init__(self):
        self._lookup_pcs = {pc: i for (i, pc) in enumerate(self.pcs)}
        self._pc_voicing_cache = {}
        self._max_doublings = defaultdict(default2)
        if self.is_consonant:
            # By default we permit tripling the root of consonant triads
            self._max_doublings[self.chord_factor_to_bass_factor[0]] = 3

    @property
    def foot(self):
        return self.pcs[0]

    def copy(self):
        """
        >>> rntxt = "m1 C: I"
        >>> (I,), _ = get_chords_from_rntxt(rntxt)
        >>> I.copy()  # doctest: +NORMALIZE_WHITESPACE
        Chord(pcs=(0, 4, 7), scale_pcs=(0, 2, 4, 5, 7, 9, 11), onset=Fraction(0, 1),
              release=Fraction(4, 1), inversion=0, token='C:I',
              intervals_above_bass=(0, 2, 4), tendencies={},
              harmony_onset=Fraction(0, 1), harmony_release=Fraction(4, 1))
        """
        return deepcopy(self)

    @cached_property
    def _voicing_prerequisites(self) -> t.Mapping[BassFactor, t.Sequence[PitchClass]]:
        """
        >>> rntxt = "m1 C: I b3 V43"
        >>> (I, V43), _ = get_chords_from_rntxt(rntxt)
        >>> I._voicing_prerequisites
        {}
        >>> V43._voicing_prerequisites  # pitch-class 5 (F) must be present for
        ... # bass-factor 2 (G) to be added
        mappingproxy({2: (5,)})
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
        return MappingProxyType(bass_factor_prerequisites)

    @cached_property
    def bass_factor_to_chord_factor(self) -> t.Tuple[int]:
        """
        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 V42"
        >>> (V7, V65, V43, V42), _ = get_chords_from_rntxt(rntxt)
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
        >>> (V7, V65, V43, V42), _ = get_chords_from_rntxt(rntxt)
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
        >>> (V7, I), _ = get_chords_from_rntxt(rntxt)
        >>> V7.cardinality
        4
        >>> I.cardinality
        3
        """
        return len(self.pcs)

    @property
    def max_doublings(self):  # pylint: disable=missing-docstring
        return self._max_doublings

    def pitch_to_bass_factor(self, pitch: PitchOrPitchClass) -> BassFactor:
        """
        >>> rntxt = "m1 C: I b3 I6"
        >>> (I, I6), _ = get_chords_from_rntxt(rntxt)
        >>> I.pitch_to_bass_factor(60)
        0
        >>> I6.pitch_to_bass_factor(60)
        2
        """
        return self._lookup_pcs[pitch % 12]

    @property
    def is_consonant(self) -> bool:
        """
        >>> rntxt = '''m1 C: I b2 IV64 b3 vi65 b4 viio7
        ... m2 I+ b2 Cad64 b3 V54 b4 V'''
        >>> (I, IV64, vi65, viio7, Iaug, Cad64, V54, V), _ = get_chords_from_rntxt(
        ...     rntxt
        ... )
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

    def get_pcs_that_can_be_added_to_existing_voicing(
        self,
        existing_voices_not_including_bass: t.Iterable[PitchOrPitchClass] = (),
        suspensions: t.Iterable[PitchOrPitchClass] = (),
    ) -> t.Tuple[PitchClass]:
        """
        `suspensions` need to be also present in `existing_voices_not_including_bass`.
        >>> rntxt = "m1 C: V7 b2 V65 b3 V42"
        >>> (V7, V65, V42), _ = get_chords_from_rntxt(rntxt)

        >>> V7.get_pcs_that_can_be_added_to_existing_voicing()
        (7, 11, 2, 5)

        >>> V7.get_pcs_that_can_be_added_to_existing_voicing((11, 2))
        (7, 2, 5)

        >>> V65.get_pcs_that_can_be_added_to_existing_voicing()
        (2, 5, 7)

        >>> V42.get_pcs_that_can_be_added_to_existing_voicing((11,))
        (7, 2)

        >>> V42.get_pcs_that_can_be_added_to_existing_voicing((0,), suspensions=(0,))
        (7, 2)
        """
        omissions = self.get_omissions(
            (self.foot,) + tuple(existing_voices_not_including_bass),
            suspensions=suspensions,
        )
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
        resolve_up_by: t.Tuple[ChromaticInterval, ...] = (1,),
    ) -> Resolution | None:
        """
        >>> rntxt = '''m1 C: I7 b3 V65/ii'''
        >>> (
        ...     I7,
        ...     V65_of_ii,
        ... ), _ = get_chords_from_rntxt(rntxt)
        >>> V65_of_ii.get_tendency_resolutions(69, Tendency.NONE)

        >>> V65_of_ii.get_tendency_resolutions(68, Tendency.DOWN)
        Resolution(by=-1, to=67)
        >>> V65_of_ii.get_tendency_resolutions(68, Tendency.UP)
        Resolution(by=1, to=69)
        >>> V65_of_ii.get_tendency_resolutions(65, Tendency.UP)

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
        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: V7
        ... m2 viio6
        ... m3 Cad64'''
        >>> (V7, viio6, Cad64), _ = get_chords_from_rntxt(rntxt)
        >>> V7.get_pitch_tendency(11)
        <Tendency.UP: 2>
        >>> viio6.get_pitch_tendency(5)
        <Tendency.DOWN: 3>
        >>> Cad64.get_pitch_tendency(0)
        <Tendency.DOWN: 3>
        """
        bass_factor = self._lookup_pcs[pitch % 12]
        return self.tendencies.get(bass_factor, Tendency.NONE)

    def pc_can_be_doubled(self, pitch_or_pc: int) -> bool:
        """
        >>> rntxt = "m1 C: V7"
        >>> (V7,), _ = get_chords_from_rntxt(rntxt)
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
        >>> (I, V7, viio7_of_vi, ii64), _ = get_chords_from_rntxt(rntxt)
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
        counts = Counter([self._lookup_pcs[pitch % 12] for pitch in pitches])
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
        # TODO: (Malcolm 2023-07-14) depending on what this function is used for,
        #   it maybe needs to account for suspensions too.
        Returns true if the pc is a tendency tone that is already present
        among the existing pitches.

        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: V7
        ... m2 I'''
        >>> (dom7, tonic), _ = get_chords_from_rntxt(rntxt)
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

    def get_omissions(
        self,
        existing_pitches_or_pcs: t.Iterable[PitchOrPitchClass],
        suspensions: t.Iterable[PitchOrPitchClass] = (),
        iq: t.Optional[IntervalQuerier] = None,
    ) -> t.List[Allow]:
        """Get pitches that can or must be omitted based on existing pitches.

        >>> rntxt = "m1 C: V7 b2 V43 b3 I"
        >>> (V7, V43, I), _ = get_chords_from_rntxt(rntxt)

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
        melody_pitch: Pitch | None = None,
        range_constraints: RangeConstraints = RangeConstraints(),
        spacing_constraints: SpacingConstraints = SpacingConstraints(),
        shuffled: bool = True,
    ) -> t.Iterable[t.Tuple[Pitch]]:
        """
        This doesn't allow specifying suspensions, etc., because it's only intended
        to get an initial chord spacing.
        >>> rntxt = "m1 C: V7 b2 V65 b3 V43 b4 I"
        >>> (V7, V65, V43, I), _ = get_chords_from_rntxt(rntxt)

        >>> voicing_iter = I.pitch_voicings()
        >>> next(voicing_iter), next(voicing_iter), next(voicing_iter)  # doctest: +SKIP
        ((72, 76, 79, 79), (60, 64, 64, 67), (48, 52, 60, 64))

        """
        pc_voicings = self.pc_voicings(
            min_notes,
            max_notes,
            max_doubling=max_doubling,
            included_factors=() if melody_pitch is None else (melody_pitch,),
        )

        spacing_iters = [
            yield_spacings(
                pcs=pc_voicing,
                range_constraints=range_constraints,
                spacing_constraints=spacing_constraints,
                melody_pitch=melody_pitch,
                shuffled=shuffled,
            )
            for pc_voicing in pc_voicings
        ]
        yield from yield_from_sequence_of_iters(spacing_iters, shuffle=True)

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
        >>> (V7, V65, V43, I), _ = get_chords_from_rntxt(rntxt)
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
        # TODO: (Malcolm) should we change the contents of `counts` in the case of a
        # bass suspension?
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
        >>> (V7, I), _ = get_chords_from_rntxt(rntxt)

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

        >>> rntxt = '''m1 C: V7 b2 V65 b3 V6 b4 V43
        ... m2 I'''
        >>> (V7, V65, V6, V43, I), _ = get_chords_from_rntxt(rntxt)

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
        >>> (dom7, viio7, i, iv), _ = get_chords_from_rntxt(rntxt)
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
        >>> (chord1, chord2, chord3), _ = get_chords_from_rntxt(rntxt)
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
        out._lookup_pcs = {pc: i for (i, pc) in enumerate(out.pcs)}
        return out


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
    >>> (I, Ib, I6, V7_of_IV, F_V), _ = get_chords_from_rntxt(rntxt)
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


def get_inversionless_figure(rn: music21.roman.RomanNumeral):
    """It seems that music21 doesn't provide a method for returning everything
    *but* the numeric figures from a roman numeral token.

    >>> RN = music21.roman.RomanNumeral
    >>> get_inversionless_figure(RN("V6"))
    'V'
    >>> get_inversionless_figure(RN("V+6"))
    'V+'
    >>> get_inversionless_figure(RN("viio6"))
    'viio'
    >>> get_inversionless_figure(RN("Cad64"))
    'Cad'
    """
    if rn.figure.startswith("Cad"):
        # Cadential 6/4 chord is a special case
        return "Cad"
    return rn.primaryFigure.rstrip("0123456789/")


def apply_tendencies(rn: music21.roman.RomanNumeral) -> t.Dict[BassFactor, Tendency]:
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
    figure = get_inversionless_figure(rn)
    if figure not in TENDENCIES:
        return {}
    raw_tendencies = TENDENCIES[figure]
    return {
        (chord_factor_i - inversion) % cardinality: raw_tendencies[chord_factor_i]
        for chord_factor_i in range(cardinality)
        if chord_factor_i in raw_tendencies
    }


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
        # TODO these special cases are hacks until music21's RomanNumeral
        #   handling is repaired or I figure out another solution
        if rn.figure == "bII7":
            scale_pcs = [p.pitchClass for p in key.pitches[:-1]]  # type:ignore
            _add_inflection(1, -1)
            _add_inflection(5, -1)
        else:
            raise
    return tuple(scale_pcs)


def _get_chord_pcs(rn: music21.roman.RomanNumeral) -> t.Tuple[PitchClass]:
    # TODO remove this function after music21's RomanNumeral
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


def voice_lead_chords(
    chord1: Chord,
    chord2: Chord,
    chord1_pitches: t.Sequence[Pitch],
    chord1_suspensions: t.Mapping[Pitch, Suspension] = MappingProxyType({}),
    chord2_melody_pitch: Pitch | None = None,
    chord2_suspensions: t.Dict[Pitch, Suspension] | None = None,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    raise_error_on_failure_to_resolve_tendencies: bool = False,
    max_diff_number_of_voices: int = 0,
    spacing_constraints: SpacingConstraints = SpacingConstraints(),
) -> t.Iterator[t.Tuple[Pitch]]:
    """Voice-lead, taking account of tendency tones, etc.

    >>> rntxt = '''m1 C: I b2 I6 b3 V6 b4 ii
    ... m2 V43 b2 V/IV b3 IV b4 V'''
    >>> (I, I6, V6, ii, V43, V_of_IV, IV, V), _ = get_chords_from_rntxt(rntxt)

    Note: we don't down-weight doubling thirds at all. When it comes to close-position
    chords this corresponds to baroque figured-bass practice, but it corresponds less
    well to other styles of harmonization. (On the other hand, see Huron's comments
    on doubling.)
    >>> vl_iter = voice_lead_chords(I, I6, (60, 64, 67, 72))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((64, 64, 67, 72), (64, 67, 67, 72), (52, 64, 67, 72))

    >>> vl_iter = voice_lead_chords(I, I6, (60, 64, 67, 72), max_bass_pitch=60)
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((52, 64, 67, 72), (52, 67, 67, 72), (52, 60, 67, 72))

    If `raise_error_on_failure_to_resolve_tendencies` is True and a tendency tone
    can't be resolved, there is a ValueError
    >>> next(
    ...     voice_lead_chords(
    ...         V6,
    ...         ii,
    ...         (59, 62, 67),  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...         raise_error_on_failure_to_resolve_tendencies=True,
    ...     )
    ... )
    Traceback (most recent call last):
    ValueError:

    >>> vl_iter = voice_lead_chords(V6, I, (47, 55, 62, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 67), (48, 52, 60, 67), (48, 55, 60, 64))

    >>> vl_iter = voice_lead_chords(V43, I, (50, 59, 65, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 60, 64, 67), (48, 60, 64, 64), (48, 60, 64, 72))

    Although the pitch-class content of I and V_of_IV is the same, the results aren't
    the same because we avoid doubling tendency tones such as the leading-tone of
    V_of_IV.
    >>> vl_iter = voice_lead_chords(V43, V_of_IV, (50, 59, 65, 67))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 60, 64, 67), (48, 60, 64, 72), (48, 60, 60, 64))

    ------------------------------------------------------------------------------------
    Suspensions in chord 1
    ------------------------------------------------------------------------------------

    Suspension w/ repetition of same harmony:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)
    >>> vl_iter = voice_lead_chords(
    ...     I, I, (48, 55, 65, 72), chord1_suspensions={65: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 72), (48, 52, 64, 72), (48, 55, 64, 76))

    Suspension w/ change of harmony:
    >>> suspension = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=10)
    >>> vl_iter = voice_lead_chords(
    ...     I6, IV, (52, 62, 67, 72), chord1_suspensions={62: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((53, 60, 69, 72), (53, 60, 69, 69), (53, 60, 65, 69))

    Suspension in bass w/ repetition of same harmony:
    >>> suspension = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)
    >>> vl_iter = voice_lead_chords(
    ...     I, I, (50, 55, 64, 67), chord1_suspensions={50: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 64, 67), (48, 55, 64, 64), (48, 52, 64, 67))

    Suspension in bass w/ change of harmony:
    >>> suspension = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)
    >>> vl_iter = voice_lead_chords(
    ...     ii, V43, (52, 53, 62, 69), chord1_suspensions={52: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 53, 62, 67), (50, 53, 62, 71), (50, 53, 59, 67))

    ------------------------------------------------------------------------------------
    Providing chord 2 melody pitch
    ------------------------------------------------------------------------------------

    Melody "overlaps" with inner voice
    >>> vl_iter = voice_lead_chords(I, V6, (48, 55, 64, 67), chord2_melody_pitch=62)
    >>> next(vl_iter), next(vl_iter)
    ((47, 55, 62, 62), (47, 55, 55, 62))

    >>> next(vl_iter)  # There are no more voice-leadings that work
    Traceback (most recent call last):
    StopIteration

    Melody with suspension in bass
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=0)
    >>> vl_iter = voice_lead_chords(
    ...     V6,
    ...     V6,
    ...     (48, 50, 55, 62),
    ...     chord2_melody_pitch=67,
    ...     chord1_suspensions={48: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((47, 50, 55, 67), (47, 50, 62, 67), (47, 62, 62, 67))

    Melody with suspension in inner part
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     I,
    ...     (48, 53, 60, 67),
    ...     chord2_melody_pitch=72,
    ...     chord1_suspensions={53: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 52, 60, 72), (48, 52, 64, 72), (52, 60, 60, 72))

    Melody with suspension in melody
    >>> suspension = Suspension(resolves_by=-2, dissonant=False, interval_above_bass=9)
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (48, 52, 60, 69),
    ...     chord2_melody_pitch=67,
    ...     chord1_suspensions={69: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((47, 50, 62, 67), (47, 55, 62, 67), (47, 50, 55, 67))

    ------------------------------------------------------------------------------------
    Suspensions in chord 2
    ------------------------------------------------------------------------------------

    Suspension in inner voice:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=10)
    >>> vl_iter = voice_lead_chords(
    ...     I, V43, (48, 55, 60, 64), chord2_suspensions={60: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 55, 60, 65), (50, 53, 60, 62), (50, 53, 60, 67))

    # TODO: (Malcolm) test more thoroughly; make sure exclude motions works in all
    # cases
    Unprepared suspension in inner voice:
    >>> suspension = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=6)
    >>> vl_iter = voice_lead_chords(
    ...     I, IV, (60, 67, 72, 76), chord2_suspensions={71: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((65, 71, 72, 77), (65, 65, 71, 72), (65, 65, 71, 77))

    Suspension in melody voice:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=10)
    >>> vl_iter = voice_lead_chords(
    ...     I, V43, (48, 55, 64, 72), chord2_suspensions={72: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((50, 55, 65, 72), (50, 53, 62, 72), (50, 62, 65, 72))

    Unprepared suspension in melody voice:
    # TODO: (Malcolm 2023-07-13) this has a bug
    # >>> suspension = Suspension(resolves_by=-2, dissonant=False, interval_above_bass=9)
    # >>> vl_iter = voice_lead_chords(I, V43, (48, 52, 60, 67), chord2_melody_pitch=64,
    # ...                             chord2_suspensions={64: suspension})
    # >>> next(vl_iter), next(vl_iter), next(vl_iter)


    Suspension overlapping with melody voice:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=10)
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V,
    ...     (48, 64, 72, 72),
    ...     chord2_melody_pitch=74,
    ...     chord2_suspensions={72: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((43, 62, 72, 74), (43, 67, 72, 74), (55, 62, 72, 74))

    Unprepared suspension overlapping with melody voice:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=10)
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V,
    ...     (48, 64, 67, 72),
    ...     chord2_melody_pitch=74,
    ...     chord2_suspensions={72: suspension},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((43, 67, 72, 74), (55, 67, 72, 74), (43, 62, 72, 74))

    Suspension in bass voice:
    >>> suspension = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=0)
    >>> vl_iter = voice_lead_chords(
    ...     I, V6, (48, 55, 64, 72), chord2_suspensions={48: suspension}
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((48, 55, 62, 74), (48, 55, 67, 74), (48, 55, 62, 67))

    Multiple inner suspensions:
    >>> suspension1 = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=10)
    >>> suspension2 = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=1)
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I6,
    ...     (50, 62, 65, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((52, 62, 65, 72), (40, 62, 65, 72))
    >>> next(
    ...     vl_iter
    ... )  # Because all 3 upper voices are fixed, this exhausts the iterator
    Traceback (most recent call last):
    StopIteration

    Melody suspension w/ inner suspension:
    >>> suspension1 = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=10)
    >>> suspension2 = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=1)
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I6,
    ...     (50, 55, 62, 65),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((52, 55, 62, 65), (40, 55, 62, 65))

    Bass suspension w/ melody suspension:
    >>> suspension1 = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)
    >>> suspension2 = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=3)
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 67, 71, 77),
    ...     chord2_suspensions={62: suspension1, 77: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter)
    ((62, 67, 72, 77), (62, 72, 72, 77))

    Bass suspension w/ inner suspension:
    >>> suspension1 = Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)
    >>> suspension2 = Suspension(resolves_by=-1, dissonant=True, interval_above_bass=3)
    >>> vl_iter = voice_lead_chords(
    ...     V43,
    ...     I,
    ...     (62, 65, 67, 71),
    ...     chord2_suspensions={62: suspension1, 65: suspension2},
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((62, 65, 67, 72), (62, 65, 72, 72), (62, 65, 72, 79))

    ------------------------------------------------------------------------------------
    Chord progression likely to lead to parallels
    ------------------------------------------------------------------------------------

    >>> vl_iter = voice_lead_chords(I, ii, (60, 64, 67, 72))
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((62, 65, 65, 69), (62, 62, 65, 69), (62, 62, 65, 77))

    ------------------------------------------------------------------------------------
    Spacing constraints
    ------------------------------------------------------------------------------------

    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (60, 64, 67, 72),
    ...     spacing_constraints=SpacingConstraints(max_adjacent_interval=5),
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((59, 62, 67, 67), (59, 67, 67, 67), (59, 62, 62, 67))

    Starting from a very widely spaced chord
    >>> vl_iter = voice_lead_chords(
    ...     I,
    ...     V6,
    ...     (36, 48, 64, 79),
    ...     spacing_constraints=SpacingConstraints(max_adjacent_interval=9),
    ... )
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((35, 55, 62, 67), (35, 62, 67, 74), (35, 55, 62, 62))

    ------------------------------------------------------------------------------------
    Different numbers of voices
    ------------------------------------------------------------------------------------

    # TODO: (Malcolm) omitted voices do not contribute to the voice-leading displacement
    # and so fewer numbers of voices are always favored over greater numbers of voices.
    # What is the right way of addressing this?
    #   - with a numeric penalty each time a voice is dropped?
    >>> vl_iter = voice_lead_chords(I, V6, (60, 64, 67), max_diff_number_of_voices=1)
    >>> next(vl_iter), next(vl_iter), next(vl_iter)
    ((59, 67), (59, 62), (59, 62, 67))
    """

    chord2_max_notes = len(chord1_pitches) + max_diff_number_of_voices
    chord2_min_notes = len(chord1_pitches) - max_diff_number_of_voices

    chord1_melody_pitch = max(chord1_pitches)

    bass_suspension = None
    melody_suspension = False
    if chord2_suspensions:
        # TODO: should this be an argument? attribute of each suspension? should it
        #   just be removed, and up to the caller to prepare?
        enforce_preparations = False
        if enforce_preparations:  # TODO: (Malcolm)
            assert all(p in chord1_pitches for p in chord2_suspensions)

        for pitch, suspension in chord2_suspensions.items():
            if suspension.interval_above_bass == 0:
                bass_suspension = pitch
            elif pitch == chord1_melody_pitch:
                if chord2_melody_pitch is None:
                    chord2_melody_pitch = pitch
                    melody_suspension = True
                elif pitch == chord2_melody_pitch:
                    melody_suspension = True
        if bass_suspension is not None:
            chord2_suspensions.pop(bass_suspension)
    elif chord2_suspensions is None:
        chord2_suspensions = {}

    resolution_pitches = []
    unresolved_suspensions = []
    unresolved_tendencies = []
    pitches_without_tendencies = []
    for i, pitch in enumerate(
        chord1_pitches[: (None if chord2_melody_pitch is None else -1)]
    ):
        # ------------------------------------------------------------------------------
        # Case 1 pitch is suspension in chord 1
        # ------------------------------------------------------------------------------
        if pitch in chord1_suspensions:
            resolution_pitch = pitch + chord1_suspensions[pitch].resolves_by
            if resolution_pitch % 12 in chord2.pcs:
                if i != 0 and pitch not in chord2_suspensions:
                    # we don't include the bass among the resolution pitches because it
                    # is always already included
                    resolution_pitches.append(resolution_pitch)
            else:
                unresolved_suspensions.append(pitch)
        # ------------------------------------------------------------------------------
        # Case 2 pitch has tendency
        # ------------------------------------------------------------------------------
        elif (pitch_tendency := chord1.get_pitch_tendency(pitch)) is not Tendency.NONE:
            resolution = chord2.get_tendency_resolutions(pitch, pitch_tendency)
            if resolution is not None:
                if i != 0 and pitch not in chord2_suspensions:
                    # we don't include the bass among the resolution pitches because it
                    # is always already included
                    resolution_pitches.append(resolution.to)
            else:
                unresolved_tendencies.append(pitch)

        # ------------------------------------------------------------------------------
        # Case 1 pitch is not suspension and has no tendency
        # ------------------------------------------------------------------------------
        else:
            if i != 0 and pitch not in chord2_suspensions:
                pitches_without_tendencies.append(pitch)

    if unresolved_suspensions:
        raise ValueError("Suspensions cannot resolve")
    if raise_error_on_failure_to_resolve_tendencies and unresolved_tendencies:
        raise ValueError("Tendencies cannot resolve")

    # TODO: (Malcolm) do we care that chord2_melody_pitch can potentially double
    #   a member of resolution_pitches?
    prespecified_pitches = tuple(resolution_pitches) + (
        (chord2_melody_pitch,)
        if (chord2_melody_pitch is not None and not melody_suspension)
        else ()
    )

    chord2_options = chord2.get_pcs_needed_to_complete_voicing(
        other_chord_factors=prespecified_pitches,
        suspensions=chord2_suspensions,
        bass_suspension=bass_suspension,
        min_notes=chord2_min_notes,
        max_notes=chord2_max_notes,
    )

    whole_tone_suspension_resolution_pitches = [
        p - 2 for (p, s) in chord2_suspensions.items() if s.resolves_by == -2
    ]

    pitches_to_voice_lead_from = (
        [chord1_pitches[0]] if bass_suspension is None else []
    ) + pitches_without_tendencies

    chord2_suspension_pitches = (
        () if bass_suspension is None else (bass_suspension,)
    ) + tuple(chord2_suspensions)

    if chord2_melody_pitch is not None:
        if max_pitch is None:
            max_pitch = chord2_melody_pitch
        else:
            max_pitch = min(chord2_melody_pitch, max_pitch)

    if bass_suspension is not None:
        if min_pitch is None:
            min_pitch = bass_suspension + 1
        else:
            min_pitch = max(bass_suspension + 1, min_pitch)

    exclude_motions = {
        i: [r - p for r in whole_tone_suspension_resolution_pitches]
        for i, p in enumerate(pitches_to_voice_lead_from)
    }

    for (
        candidate_pitches,
        voice_assignments,
    ) in voice_lead_pitches_multiple_options_iter(
        pitches_to_voice_lead_from,
        [
            ([chord2.pcs[0]] if bass_suspension is None else []) + option
            for option in chord2_options
        ],
        # We handle the bass separately if there is a bass suspension
        preserve_bass=bass_suspension is None,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        min_bass_pitch=min_bass_pitch,
        max_bass_pitch=max_bass_pitch,
        exclude_motions=exclude_motions,
    ):
        output = tuple(
            sorted(candidate_pitches + prespecified_pitches + chord2_suspension_pitches)
        )
        if not validate_spacing(output, spacing_constraints):
            continue
        if succession_has_forbidden_parallels(chord1_pitches, output):
            continue
        if outer_voices_have_forbidden_antiparallels(chord1_pitches, output):
            continue
        yield output


@cacher()
def get_chords_from_rntxt(
    rn_data: str,
    split_chords_at_metric_strong_points: bool = True,
    no_added_tones: bool = True,
) -> t.Union[
    t.Tuple[t.List[Chord], Meter, music21.stream.Score],  # type:ignore
    t.Tuple[t.List[Chord], Meter],
]:
    """Converts roman numerals to pcs.

    Args:
        rn_data: either path to a romantext file or the contents thereof.
    """
    if no_added_tones:
        rn_data = strip_added_tones(rn_data)
    score = parse_rntxt(rn_data)
    m21_ts = score[music21.meter.TimeSignature].first()  # type:ignore
    ts = f"{m21_ts.numerator}/{m21_ts.denominator}"  # type:ignore
    ts = Meter(ts)
    prev_scale = duration = start = pickup_offset = key = None
    prev_chord = None
    out_list = []
    for rn in score.flatten()[music21.roman.RomanNumeral]:
        if pickup_offset is None:
            pickup_offset = TIME_TYPE(((rn.beat) - 1) * rn.beatDuration.quarterLength)
        chord = _get_chord_pcs(rn)
        scale = fit_scale_to_rn(rn)
        if scale != prev_scale or chord != prev_chord.pcs:  # type:ignore
            if prev_chord is not None:
                out_list.append(prev_chord)
            start = TIME_TYPE(rn.offset) + pickup_offset
            duration = TIME_TYPE(rn.duration.quarterLength)
            intervals_above_bass = tuple(
                (scale.index(pc) - scale.index(chord[0])) % len(scale) for pc in chord
            )
            tendencies = apply_tendencies(rn)
            if rn.key.tonicPitchNameWithCase != key:  # type:ignore
                key = rn.key.tonicPitchNameWithCase  # type:ignore
                pre_token = key + ":"
            else:
                pre_token = ""
            prev_chord = Chord(
                pcs=chord,
                scale_pcs=scale,
                onset=start,
                release=start + duration,
                inversion=rn.inversion(),
                token=pre_token + rn.figure,
                intervals_above_bass=intervals_above_bass,
                tendencies=tendencies,
            )
            prev_scale = scale
        else:
            prev_chord.release += TIME_TYPE(rn.duration.quarterLength)  # type:ignore

    out_list.append(prev_chord)

    if split_chords_at_metric_strong_points:
        chord_list = ts.split_at_metric_strong_points(
            out_list, min_split_dur=ts.beat_dur
        )
        out_list = []
        for chord in chord_list:
            out_list.extend(ts.split_odd_duration(chord, min_split_dur=ts.beat_dur))
    get_harmony_onsets_and_releases(out_list)
    return out_list, ts


# doctests in cached_property methods are not discovered and need to be
#   added explicitly to __test__; see https://stackoverflow.com/a/72500890/10155119
__test__ = {
    "Chord.augmented_second_adjustments": Chord.augmented_second_adjustments,
    "Chord.bass_factor_to_chord_factor": Chord.bass_factor_to_chord_factor,
    "Chord.chord_factor_to_bass_factor": Chord.chord_factor_to_bass_factor,
    "Chord._voicing_prerequisites": Chord._voicing_prerequisites,
}
