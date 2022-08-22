from numbers import Number
import music21
import pandas as pd

import typing as t

from dumb_composer.constants import TIME_TYPE


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


def get_chords_from_rntxt(rn_data: str) -> t.Tuple[pd.DataFrame, float, str]:
    """Converts roman numerals to pcs.

    Args:
        rn_data: string in romantext format.
    Returns:
        A t.Tuple (DataFrame, float, str). The float
            indicates the offset for a pickup measure
            (0 if there is no pickup measure). The string
            represents the time signature (e.g., "4/4").
    """
    score = music21.converter.parse(rn_data, format="romanText").flatten()
    ts = f"{score.timeSignature.numerator}/{score.timeSignature.denominator}"
    prev_scale = duration = start = pickup_offset = key = None
    prev_chord = None
    out_list = []
    for rn in score[music21.roman.RomanNumeral]:
        if pickup_offset is None:
            pickup_offset = (
                TIME_TYPE(rn.beat) - 1
            ) * rn.beatDuration.quarterLength
        chord = tuple(rn.pitchClasses)
        # TODO scale will not contain any alterations that are in the chord!
        # music21 returns the scale *including the upper octave*, which we do
        #   not want
        scale = tuple(p.pitchClass for p in rn.key.pitches[:-1])
        if scale != prev_scale or chord != prev_chord.pcs:
            if prev_chord is not None:
                out_list.append(prev_chord)
            start = TIME_TYPE(rn.offset)
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
    # out_df = pd.DataFrame(
    #     out_list,
    # )
    return out_list, pickup_offset, ts
