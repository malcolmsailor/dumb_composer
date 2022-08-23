from numbers import Number
import music21
import pandas as pd

import typing as t

from dumb_composer.constants import TIME_TYPE
from dumb_composer.time import Meter


class Chord(pd.Series):
    def __init__(
        self,
        pcs: t.Tuple[int],
        scale_pcs: t.Tuple[int],
        onset: Number,
        release: Number,
        inversion: int,
        token: str,
        intervals_above_bass: t.Tuple[int],
    ):
        foot = pcs[0]
        super().__init__(
            {
                "pcs": pcs,
                "scale_pcs": scale_pcs,
                "onset": onset,
                "release": release,
                "foot": foot,
                "inversion": inversion,
                "token": token,
                "intervals_above_bass": intervals_above_bass,
            }
        )


def fit_scale_to_rn(rn: music21.roman.RomanNumeral) -> t.Tuple[int]:
    """
    >>> RN = music21.roman.RomanNumeral
    >>> fit_scale_to_rn(RN("viio7", keyOrScale="C")) # Note A-flat
    (0, 2, 4, 5, 7, 8, 11)
    >>> fit_scale_to_rn(RN("Ger6", keyOrScale="C")) # Note A-flat, E-flat, F-sharp
    (0, 2, 3, 6, 7, 8, 11)


    If the roman numeral has a secondary key, we use that as the scale.
    TODO I'm not sure this is always desirable.

    >>> RN = music21.roman.RomanNumeral
    >>> fit_scale_to_rn(RN("V/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 4, 6)
    >>> fit_scale_to_rn(RN("viio7/V", keyOrScale="C"))
    (7, 9, 11, 0, 2, 3, 6)
    >>> fit_scale_to_rn(RN("viio7/bIII", keyOrScale="C")) # Note C-flat
    (3, 5, 7, 8, 10, 11, 2)
    """

    def _add_inflection(degree: int, inflection: int):
        scale_pcs[degree] = (scale_pcs[degree] + inflection) % 12

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


def get_chords_from_rntxt(
    rn_data: str, split_chords_at_metric_strong_points: bool = True
) -> t.Tuple[t.List[Chord], float, Meter]:
    """Converts roman numerals to pcs.

    Args:
        rn_data: either path to a romantext file or the contents thereof.
    Returns:
        Tuple of:
            - list of Chord
            - float indicating length of pickup (0 if no pickup)
            - Meter
    """
    score = music21.converter.parse(rn_data, format="romanText").flatten()
    ts = f"{score.timeSignature.numerator}/{score.timeSignature.denominator}"
    ts = Meter(ts)
    prev_scale = duration = start = pickup_offset = key = None
    prev_chord = None
    out_list = []
    for rn in score[music21.roman.RomanNumeral]:
        if pickup_offset is None:
            pickup_offset = (
                TIME_TYPE(rn.beat) - 1
            ) * rn.beatDuration.quarterLength
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
            )
            prev_scale = scale
        else:
            prev_chord.release += rn.duration.quarterLength

    out_list.append(prev_chord)
    if split_chords_at_metric_strong_points:
        out_list = ts.split_at_metric_strong_points(out_list)
    return out_list, pickup_offset, ts
