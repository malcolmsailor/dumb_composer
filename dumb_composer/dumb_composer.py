import music21
import pandas as pd

from typing import Optional, Sequence, Tuple, Union

from .chord_spacer import ChordSpacer
from .patterns import PatternMaker
from .constants import TIME_TYPE


def pc_int(pc1: int, pc2: int, tet: int = 12) -> int:
    return (pc2 - pc1) % tet


def pitch_int(p1: int, p2: int) -> int:
    raise NotImplementedError


def transpose_pc_eq(
    pcs1: Sequence[int], pcs2: Sequence[int], tet: int = 12
) -> bool:
    if len(pcs1) != len(pcs2):
        return False
    for (pc1a, pc1b), (pc2a, pc2b) in zip(
        zip(pcs1[:-1], pcs1[:1]), zip(pcs2[:-1], pcs2[:1])
    ):
        if pc_int(pc1a, pc1b, tet=tet) != pc_int(pc2a, pc2b, tet=tet):
            return False
    return True


def reduce_num_pcs(pcs: Tuple[int]):
    pass


class Annotation(pd.Series):
    def __init__(self, onset, text):
        super().__init__({"onset": onset, "text": text, "type": "text"})


class DumbComposer:
    def __init__(
        self,
        pattern=None,
        text_annotations: Union[bool, str, Sequence[str]] = False,
    ):
        self._pattern = pattern
        self._text_annotations = text_annotations
        if isinstance(text_annotations, str):
            text_annotations = (text_annotations,)

    def __call__(
        self,
        chord_data: Union[str, pd.DataFrame],
        ts: str = None,
        text_annotations: Optional[Union[bool, str, Sequence[str]]] = None,
    ):
        """If chord_data is a string, it should be a roman text file.
        If it is a dataframe, it should be in the same format as returned
        by rn_to_pc
        """
        if text_annotations is None:
            text_annotations = self._text_annotations

        if isinstance(chord_data, str):
            chord_data, _, ts = rn_to_pc(chord_data)
        pm = PatternMaker(ts)
        cs = ChordSpacer()
        notes = []
        for _, chord in chord_data.iterrows():
            pitches = cs(chord.pcs)
            pattern_notes = pm(
                pitches,
                chord.onset,
                release=chord.release,
                pattern=self._pattern,
            )
            if text_annotations:
                annotations = []
                if (
                    isinstance(text_annotations, bool)
                    or "pattern" in text_annotations
                ):
                    annotations.append(pm.prev_pattern)
                if (
                    isinstance(text_annotations, bool)
                    or "chord" in text_annotations
                ):
                    annotations.append(chord.token)
                notes.append(Annotation(chord.onset, " ".join(annotations)))
            notes.extend(pattern_notes)
        out_df = pd.DataFrame(notes)
        return out_df


def rn_to_pc(rn_data: str) -> Tuple[pd.DataFrame, float, str]:
    """Converts roman numerals to pcs.

    Args:
        rn_data: string in romantext format.
    Returns:
        A tuple (DataFrame, float, str). The float
            indicates the offset for a pickup measure
            (0 if there is no pickup measure). The string
            represents the time signature (e.g., "4/4").
    """
    score = music21.converter.parse(rn_data, format="romanText").flatten()
    ts = f"{score.timeSignature.numerator}/{score.timeSignature.denominator}"
    prev_chord = duration = start = pickup_offset = key = None
    out_list = []
    for rn in score.getElementsByClass(music21.roman.RomanNumeral):
        if pickup_offset is None:
            pickup_offset = (
                TIME_TYPE(rn.beat) - 1
            ) * rn.beatDuration.quarterLength
        chord = tuple(rn.pitchClasses)
        if chord != prev_chord:
            if prev_chord is not None:
                out_list.append(
                    (
                        prev_chord,
                        start,
                        start + duration,
                        prev_chord[0],
                        rn.inversion(),
                        pre_token + prev_figure,
                    )
                )
            start = TIME_TYPE(rn.offset)
            duration = TIME_TYPE(rn.duration.quarterLength)
        else:
            duration += rn.duration.quarterLength
        prev_chord = chord
        if rn.key != key:
            key = rn.key.tonicPitchNameWithCase
            pre_token = key + ":"
        else:
            pre_token = ""
        prev_figure = rn.figure
    out_list.append(
        (
            prev_chord,
            start,
            start + duration,
            prev_chord[0],
            rn.inversion(),
            pre_token + prev_figure,
        )
    )
    out_df = pd.DataFrame(
        out_list,
        columns=["pcs", "onset", "release", "root", "inversion", "token"],
    )
    return out_df, pickup_offset, ts
