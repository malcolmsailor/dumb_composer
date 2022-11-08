from __future__ import annotations

from copy import deepcopy
import copy
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Number
import os
import re
from types import MappingProxyType
import music21
import pandas as pd

import typing as t

from dumb_composer.constants import TIME_TYPE, speller_pcs, unspeller_pcs
from cache_lib import cacher
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.music21_handler import parse_rntxt
from dumb_composer.time import Meter
from enum import Enum, auto


class Allow(Enum):
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


TENDENCIES = MappingProxyType(
    {
        "V": {1: Tendency.UP, 3: Tendency.DOWN},
        "viio": {0: Tendency.UP, 2: Tendency.DOWN, 3: Tendency.DOWN},
        "Ger": {0: Tendency.UP, 1: Tendency.DOWN},
        "Fr": {1: Tendency.UP, 2: Tendency.DOWN},
        "iv": {1: Tendency.DOWN},
        # To get the correct tendencies for the cadential 64 chord we need
        #   to index into it as I
        "Cad": {0: Tendency.DOWN, 1: Tendency.DOWN},
    }
)


@dataclass
class Chord:
    pcs: t.Tuple[int]
    scale_pcs: t.Tuple[int]
    onset: Number
    release: Number
    inversion: int
    token: str
    intervals_above_bass: t.Tuple[int]
    tendencies: t.Dict[int, Tendency]

    # whereas 'onset' and 'release' should be the start of this particular
    #   structural "unit" (which might, for example, break at a barline
    #   without a change of harmony), `harmony_onset` and `harmony_release`
    #   are for the onset and release of the harmony (i.e., the boundaries
    #   of the preceding and succeeding *changes* in the harmony)
    harmony_onset: t.Optional[Number] = field(default=None, compare=False)
    harmony_release: t.Optional[Number] = field(default=None, compare=False)

    foot: t.Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        self.foot = self.pcs[0]
        self._lookup_pcs = {pc: i for (i, pc) in enumerate(self.pcs)}

    def copy(self):
        return deepcopy(self)

    def get_pitch_tendency(self, pitch: int) -> Tendency:
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
        pc_i = self._lookup_pcs[pitch % 12]
        return self.tendencies.get(pc_i, Tendency.NONE)

    def pc_must_be_omitted(
        self, pc: int, existing_pitches: t.Sequence[int]
    ) -> bool:
        """
        Returns true if the pc is a tendency tone that is already present
        among the existing pitches.

        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: V7
        ... m2 I'''
        >>> (dom7, tonic), _ = get_chords_from_rntxt(rntxt)
        >>> dom7.pc_must_be_omitted(11, [62, 71]) # B already present in chord
        True
        >>> dom7.pc_must_be_omitted(5, [62, 71]) # F not present in chord
        False
        >>> not any(tonic.pc_must_be_omitted(pc, [60, 64, 67])
        ...         for pc in (0, 4, 7)) # tonic has no tendency tones
        True

        """
        return self.get_pitch_tendency(pc) is not Tendency.NONE and any(
            pitch % 12 == pc for pitch in existing_pitches
        )

    def get_omissions(
        self,
        existing_pitches: t.Tuple[int],
        suspensions: t.Tuple[int],
        iq: IntervalQuerier,
    ) -> t.List[Allow]:
        out = []
        semitone_resolutions = {(s - 1) % 12 for s in suspensions}
        wholetone_resolutions = {(s - 2) % 12 for s in suspensions}
        for pc in self.pcs:
            if pc in semitone_resolutions or self.pc_must_be_omitted(
                pc, existing_pitches
            ):
                out.append(Allow.ONLY)
            elif pc in wholetone_resolutions or iq.pc_can_be_omitted(
                pc, existing_pitches
            ):
                out.append(Allow.YES)
            else:
                out.append(Allow.NO)
        return out

    @cached_property
    def augmented_second_adjustments(self) -> t.Dict[int, Inflection]:
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
        >>> dom7.augmented_second_adjustments # ^6 should be raised
        {5: <Inflection.UP: 3>, 6: <Inflection.NONE: 2>}
        >>> viio7.augmented_second_adjustments # both ^6 and ^7 are chord tones
        {5: <Inflection.NONE: 2>, 6: <Inflection.NONE: 2>}
        >>> i.augmented_second_adjustments # no augmented 2nd
        {}
        >>> i.scale_pcs = (9, 11, 0, 2, 4, 5, 8) # harmonic-minor scale
        >>> del i.augmented_second_adjustments # rebuild augmented_second_adjustments
        >>> i.augmented_second_adjustments
        {5: <Inflection.EITHER: 1>, 6: <Inflection.EITHER: 1>}
        >>> iv.scale_pcs = (9, 11, 0, 2, 4, 5, 8) # harmonic-minor scale
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

    def transpose(self, interval: int) -> Chord:
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
        out.foot = (out.foot + interval) % 12
        if ":" in out.token:
            # we need to transpose the key symbol
            m = re.match(r"(?P<key>[^:]+):(?P<numeral>.*)", out.token)
            key = speller_pcs((unspeller_pcs(m.group("key")) + interval) % 12)
            out.token = key + ":" + m.group("numeral")
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
    >>> chords, _ = get_chords_from_rntxt(rntxt)
    >>> is_same_harmony(chords[0], chords[1])
    True
    >>> is_same_harmony(chords[1], chords[2])
    False
    >>> is_same_harmony(chords[1], chords[2], compare_inversions=False)
    True
    >>> is_same_harmony(chords[0], chords[3], allow_subsets=True)
    False
    >>> is_same_harmony(chords[0], chords[3],
    ...     compare_scales=False, allow_subsets=True)
    True
    >>> is_same_harmony(chords[0], chords[4], compare_scales=False)
    True
    >>> is_same_harmony(chords[0], chords[4], compare_scales=True)
    False
    >>> is_same_harmony(chords[3], chords[4], allow_subsets=True)
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


def apply_tendencies(rn: music21.roman.RomanNumeral) -> t.Dict[int, Tendency]:
    """
    Keys of returned dict are indices into "inverted pcs" (i.e., the pcs in
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
        (i - inversion) % cardinality: raw_tendencies[i]
        for i in range(cardinality)
        if i in raw_tendencies
    }


def fit_scale_to_rn(rn: music21.roman.RomanNumeral) -> t.Tuple[int]:
    """
    >>> RN = music21.roman.RomanNumeral
    >>> fit_scale_to_rn(RN("viio7", keyOrScale="C")) # Note A-flat
    (0, 2, 4, 5, 7, 8, 11)
    >>> fit_scale_to_rn(RN("Ger6", keyOrScale="C")) # Note A-flat, E-flat, F-sharp
    (0, 2, 3, 6, 7, 8, 11)


    If the roman numeral has a secondary key, we use that as the scale.
    TODO I'm not sure this is always desirable.

    >>> fit_scale_to_rn(RN("V/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 4, 6)
    >>> fit_scale_to_rn(RN("viio7/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 3, 6)
    >>> fit_scale_to_rn(RN("viio7/bIII", keyOrScale="C")) # Note C-flat
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

    def _add_inflection(degree: int, inflection: int):
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
    scale_pcs = [p.pitchClass for p in key.pitches[:-1]]
    for pitch in rn.pitches:
        # NB degrees are 1-indexed so we must subtract 1 below
        degree, accidental = key.getScaleDegreeAndAccidentalFromPitch(pitch)
        if accidental is not None:
            inflection = int(accidental.alter)
            _add_inflection(degree - 1, inflection)
    try:
        assert len(scale_pcs) == len(set(scale_pcs))
    except AssertionError:
        # TODO these special cases are hacks until music21's RomanNumeral
        #   handling is repaired or I figure out another solution
        if rn.figure == "bII7":
            scale_pcs = [p.pitchClass for p in key.pitches[:-1]]
            _add_inflection(1, -1)
            _add_inflection(5, -1)
        else:
            raise
    return tuple(scale_pcs)


def _get_chord_pcs(rn: music21.roman.RomanNumeral) -> t.Tuple[int]:
    # TODO remove this function after music21's RomanNumeral
    #   handling is repaired or I figure out another solution
    def _transpose(pcs, tonic_pc):
        return tuple((pc + tonic_pc) % 12 for pc in pcs)

    if rn.figure == "bII7":
        return _transpose((1, 5, 8, 0), rn.key.tonic.pitchClass)
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
) -> t.Union[
    t.Tuple[t.List[Chord], Meter, music21.stream.Score],
    t.Tuple[t.List[Chord], Meter],
]:
    """Converts roman numerals to pcs.

    Args:
        rn_data: either path to a romantext file or the contents thereof.
    """
    if no_added_tones:
        rn_data = strip_added_tones(rn_data)
    score = parse_rntxt(rn_data)
    m21_ts = score[music21.meter.TimeSignature].first()
    ts = f"{m21_ts.numerator}/{m21_ts.denominator}"
    ts = Meter(ts)
    prev_scale = duration = start = pickup_offset = key = None
    prev_chord = None
    out_list = []
    for rn in score.flatten()[music21.roman.RomanNumeral]:
        if pickup_offset is None:
            pickup_offset = TIME_TYPE(
                ((rn.beat) - 1) * rn.beatDuration.quarterLength
            )
        chord = _get_chord_pcs(rn)
        scale = fit_scale_to_rn(rn)
        if scale != prev_scale or chord != prev_chord.pcs:
            if prev_chord is not None:
                out_list.append(prev_chord)
            start = TIME_TYPE(rn.offset) + pickup_offset
            duration = TIME_TYPE(rn.duration.quarterLength)
            intervals_above_bass = tuple(
                (scale.index(pc) - scale.index(chord[0])) % len(scale)
                for pc in chord
            )
            tendencies = apply_tendencies(rn)
            if rn.key.tonicPitchNameWithCase != key:
                key = rn.key.tonicPitchNameWithCase
                pre_token = key + ":"
            else:
                pre_token = ""
            prev_chord = Chord(
                chord,
                scale,
                start,
                start + duration,
                rn.inversion(),
                pre_token + rn.figure,
                intervals_above_bass,
                tendencies,
            )
            prev_scale = scale
        else:
            prev_chord.release += TIME_TYPE(rn.duration.quarterLength)

    out_list.append(prev_chord)

    if split_chords_at_metric_strong_points:
        chord_list = ts.split_at_metric_strong_points(
            out_list, min_split_dur=ts.beat_dur
        )
        out_list = []
        for chord in chord_list:
            out_list.extend(
                ts.split_odd_duration(chord, min_split_dur=ts.beat_dur)
            )
    get_harmony_onsets_and_releases(out_list)
    return out_list, ts


# doctests in cached_property methods are not discovered and need to be
#   added explicitly to __test__; see https://stackoverflow.com/a/72500890/10155119
__test__ = {
    "Chord.augmented_second_adjustments": Chord.augmented_second_adjustments
}
