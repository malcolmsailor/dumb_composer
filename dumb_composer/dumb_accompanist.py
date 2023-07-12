from dataclasses import dataclass
from enum import Enum, auto
import logging
import pandas as pd

import typing as t

from dumb_composer.chord_spacer import SimpleSpacer, SimpleSpacerSettings
from dumb_composer.patterns import PatternMaker
from dumb_composer.pitch_utils.chords import get_chords_from_rntxt
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.shared_classes import Annotation, Chord, Note, Score
from dumb_composer.time import Meter
from .utils.recursion import DeadEnd


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


class AccompAnnots(Enum):
    NONE = auto()
    CHORDS = auto()
    PATTERNS = auto()
    ALL = auto()


# TODO prevent changing patterns on weak beats


@dataclass
class DumbAccompanistSettings(SimpleSpacerSettings):
    pattern: t.Optional[str] = None
    accompaniment_annotations: AccompAnnots = AccompAnnots.ALL
    # provides the name of attributes of Score that the accompaniment
    #   should be kept below.
    accompaniment_below: t.Optional[t.Union[str, t.Sequence[str]]] = None
    accompaniment_above: t.Optional[t.Union[str, t.Sequence[str]]] = None
    include_bass: bool = True
    end_with_solid_chord: bool = True
    pattern_changes_on_downbeats_only: bool = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        logging.debug(f"running DumbAccompanistSettings __post_init__()")
        if isinstance(self.accompaniment_below, str):
            self.accompaniment_below = [self.accompaniment_below]


class DumbAccompanist:
    def __init__(
        self,
        settings: t.Optional[DumbAccompanistSettings] = None,
    ):
        if settings is None:
            settings = DumbAccompanistSettings()
        self.settings = settings
        self._cs = SimpleSpacer()
        self._iq = IntervalQuerier()
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

    def _get_above(self, score: Score):
        if self.settings.accompaniment_above is None:
            # TODO I think we still want to keep this above the bass
            return self.settings.accomp_range[0]
        i = len(score.accompaniments)
        return max(
            [
                max(note.pitch for note in getattr(score, name)[i])
                for name in self.settings.accompaniment_above
            ]
            + [self.settings.accomp_range[0]]
        )

    def _final_step(self, score: Score):
        if not self.settings.end_with_solid_chord:
            return self._step(score)
        i = len(score.accompaniments)
        chord = score.chords[i]
        below = self._get_below(score)
        above = self._get_above(score)
        omissions = chord.get_omissions(
            existing_pitches_or_pcs=score.get_existing_pitches(i),
            # the last step should never have a suspension
            suspensions=(),
            iq=self._iq,
        )
        for pitches in self._cs(
            chord.pcs,
            omissions=omissions,
            min_accomp_pitch=above + 1,
            max_accomp_pitch=below - 1,
            include_bass=self.settings.include_bass,
        ):
            yield [
                Note(  # type:ignore
                    pitch,
                    chord.onset,
                    chord.release,
                    track=score.accompaniments_track,
                )
                for pitch in pitches
            ]

    def _step(
        self, score: Score
    ) -> t.Iterator[t.Union[t.List[Note], t.Tuple[t.List[Note], Annotation]]]:
        assert self._pm is not None

        i = len(score.accompaniments)
        chord = score.chords[i]
        chord_change = score.is_chord_change(i)
        below = self._get_below(score)
        above = self._get_above(score)
        if i in score.suspension_indices:
            suspensions = (score.structural_melody[i],)
        else:
            suspensions = ()
        omissions = chord.get_omissions(
            # LONGTERM is there anything besides structural_bass and
            #   structural_melody to be included in omissions?
            existing_pitches_or_pcs=score.get_existing_pitches(i),
            suspensions=suspensions,
            iq=self._iq,
        )
        pattern = self._pm.get_pattern(
            chord.pcs,
            chord.onset,
            chord.harmony_onset,
            chord.harmony_release,
            pattern=self.settings.pattern,
        )
        spacing_constraints = self._pm.get_spacing_constraints(pattern)
        for pitches in self._cs(
            chord.pcs,
            omissions=omissions,
            min_accomp_pitch=above + 1,
            max_accomp_pitch=below - 1,
            include_bass=self.settings.include_bass,
            spacing_constraints=spacing_constraints,
        ):
            accompaniment_pattern = self._pm(
                pitches,
                chord.onset,
                release=chord.release,
                harmony_onset=chord.harmony_onset,
                harmony_release=chord.harmony_release,
                pattern=pattern,
                track=score.accompaniments_track,
                chord_change=chord_change,
            )
            if self.settings.accompaniment_annotations is AccompAnnots.NONE:
                yield accompaniment_pattern
            else:
                annotations = []
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.PATTERNS,
                    AccompAnnots.ALL,
                ):
                    annotations.append(Annotation(chord.onset, self._pm.prev_pattern))
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.CHORDS,
                    AccompAnnots.ALL,
                ):
                    annotations.append(Annotation(chord.onset, chord.token))
                yield (accompaniment_pattern, *annotations)
        raise DeadEnd()

    def init_new_piece(self, ts: t.Union[str, Meter]):
        self._pm = PatternMaker(
            ts,
            include_bass=self.settings.include_bass,
            pattern_changes_on_downbeats_only=self.settings.pattern_changes_on_downbeats_only,
        )

    def __call__(
        self,
        chord_data: t.Optional[t.Union[str, t.List[Chord]]] = None,
        score: t.Optional[Score] = None,
        ts: t.Optional[str] = None,
    ) -> pd.DataFrame:
        """If chord_data is a string, it should be a roman text file.
        If it is a list, it should be in the same format as returned
        by get_chords_from_rntxt

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
                chord_data, ts = get_chords_from_rntxt(chord_data)  # type:ignore
            score = Score(chord_data, ts=ts)  # type:ignore

        assert ts is not None

        self.init_new_piece(ts)
        for _ in range(len(score.chords)):
            result = next(self._step(score))
            if self.settings.accompaniment_annotations is not AccompAnnots.NONE:
                it = iter(result)
                score.accompaniments.append(next(it))  # type:ignore
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.ALL,
                    AccompAnnots.PATTERNS,
                ):
                    score.annotations["patterns"].append(next(it))  # type:ignore
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.ALL,
                    AccompAnnots.CHORDS,
                ):
                    score.annotations["chords"].append(next(it))  # type:ignore
            else:
                score.accompaniments.append(result)  # type:ignore
        return score.get_df("accompaniments")
