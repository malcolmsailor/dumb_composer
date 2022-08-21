from dataclasses import dataclass
import pandas as pd

import typing as t

from dumb_composer.chord_spacer import SimpleSpacer, SimpleSpacerSettings
from dumb_composer.patterns import PatternMaker
from dumb_composer.pitch_utils.rn_to_pc import rn_to_pc
from dumb_composer.shared_classes import Annotation, Note, Score


# def pc_int(pc1: int, pc2: int, tet: int = 12) -> int:
#     return (pc2 - pc1) % tet

# Not sure what purpose this function was intended for
# def transpose_pc_eq(
#     pcs1: Sequence[int], pcs2: Sequence[int], tet: int = 12
# ) -> bool:
#     if len(pcs1) != len(pcs2):
#         return False
#     for (pc1a, pc1b), (pc2a, pc2b) in zip(
#         zip(pcs1[:-1], pcs1[:1]), zip(pcs2[:-1], pcs2[:1])
#     ):
#         if pc_int(pc1a, pc1b, tet=tet) != pc_int(pc2a, pc2b, tet=tet):
#             return False
#     return True


@dataclass
class DumbAccompanistSettings(SimpleSpacerSettings):
    pattern: t.Optional[str] = None
    text_annotations: t.Union[bool, str, t.Sequence[str]] = False
    # provides the name of attributes of Score that the accompaniment
    #   should be kept below. (TODO: also allow accompaniment to be above
    #   things.)
    accompaniment_below: t.Optional[t.Union[str, t.Sequence[str]]] = None

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        if isinstance(self.accompaniment_below, str):
            self.accompaniment_below = [self.accompaniment_below]
        if isinstance(self.text_annotations, str):
            self.text_annotations = (self.text_annotations,)


class DumbAccompanist:
    def __init__(
        self,
        settings: t.Optional[DumbAccompanistSettings] = None,
    ):
        if settings is None:
            settings = DumbAccompanistSettings()
        self.settings = settings
        self._cs = SimpleSpacer()
        self._pm: t.Optional[PatternMaker] = None

    def _get_below(self, score: Score):
        if self.settings.accompaniment_below is None:
            return self.settings.accomp_range[1]
        i = len(score.accompaniments)
        return min(
            [
                min(note.pitch for note in getattr(score, name)[i])
                for name in self.settings.accompaniment_below
            ]
            + [self.settings.accomp_range[1]]
        )

    def _step(
        self, score: Score
    ) -> t.Union[t.List[Note], t.Tuple[t.List[Note], Annotation]]:
        i = len(score.accompaniments)
        chord = score.chords.iloc[i]

        # below = min(
        #     min(note.pitch for note in score.prefabs[i]),
        #     self.settings.accomp_range[1],
        # )
        below = self._get_below(score)
        for pitches in self._cs(chord.pcs, max_accomp_pitch=below - 1):
            accompaniment_pattern = self._pm(
                pitches,
                chord.onset,
                release=chord.release,
                pattern=self.settings.pattern,
                track=score.accompaniments_track,
            )
            if not self._text_annotations_for_this_piece:
                yield accompaniment_pattern

            annotations = []
            if (
                isinstance(self._text_annotations_for_this_piece, bool)
                or "pattern" in self._text_annotations_for_this_piece
            ):
                annotations.append(self._pm.prev_pattern)
            if (
                isinstance(self._text_annotations_for_this_piece, bool)
                or "chord" in self._text_annotations_for_this_piece
            ):
                annotations.append(chord.token)
            annotation = Annotation(chord.onset, " ".join(annotations))
            yield accompaniment_pattern, annotation

    def init_new_piece(
        self,
        ts: str,
        text_annotations: t.Optional[
            t.Union[bool, str, t.Sequence[str]]
        ] = None,
    ):
        self._pm = PatternMaker(ts)
        if text_annotations is None:
            self._text_annotations_for_this_piece = (
                self.settings.text_annotations
            )
        else:
            self._text_annotations_for_this_piece = text_annotations

    def __call__(
        self,
        chord_data: t.Optional[t.Union[str, pd.DataFrame]] = None,
        score: t.Optional[Score] = None,
        ts: t.Optional[str] = None,
        text_annotations: t.Optional[
            t.Union[bool, str, t.Sequence[str]]
        ] = None,
    ) -> pd.DataFrame:
        """If chord_data is a string, it should be a roman text file.
        If it is a dataframe, it should be in the same format as returned
        by rn_to_pc

        Keyword args:
            ts: time signature. If chord_data is a string, this argument is
                ignored.

        Returns:
            dataframe with columns onset, text, type, pitch, release
        """
        if chord_data is None and score is None:
            raise ValueError("either chord_data or score must not be None")
        if score is None:
            if isinstance(chord_data, str):
                chord_data, _, ts = rn_to_pc(chord_data)
            score = Score(chord_data)

        self.init_new_piece(ts, text_annotations)
        for _ in range(len(score.chords)):
            result = next(self._step(score))
            if self._text_annotations_for_this_piece:
                pattern, annotation = result
                score.accompaniments.append(pattern)
                score.annotations.append(annotation)
            else:
                score.accompaniments.append(result)
        return score.get_df("accompaniments")
