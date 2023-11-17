import logging
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from itertools import chain

import pandas as pd

from dumb_composer.chord_spacer import SimpleSpacer, SimpleSpacerSettings
from dumb_composer.chords.chords import get_chords_from_rntxt
from dumb_composer.classes.chord_transition_interfaces import AccompanimentInterface
from dumb_composer.classes.scores import ScoreWithAccompaniments
from dumb_composer.incremental_contrapuntist import IncrementalResult
from dumb_composer.patterns import PatternMaker
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.music21_handler import get_ts_from_rntxt
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import (
    ALTO,
    BASS,
    MELODY,
    TENOR,
    FourPartResult,
    InnerVoice,
    OuterVoice,
    Pitch,
    RecursiveWorker,
    SettingsBase,
    TwoPartResult,
    Voice,
)
from dumb_composer.shared_classes import Annotation, Note
from dumb_composer.time import Meter
from dumb_composer.utils.recursion import DeadEnd, recursive_attempt


class AccompanimentDeadEnd(DeadEnd):
    pass


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
    # provides the name of attributes of PrefabScore that the accompaniment
    #   should be kept below.
    accompaniment_below: t.Optional[t.Union[str, t.Sequence[str]]] = None
    accompaniment_above: t.Optional[t.Union[str, t.Sequence[str]]] = None
    # include_bass: bool = True
    end_with_solid_chord: bool = True
    pattern_changes_on_downbeats_only: bool = True
    range_constraints: RangeConstraints = RangeConstraints()
    pattern_inertia: float = 5.0

    # (Malcolm 2023-11-14) I'm keeping use_chord_spacer around to preserve old behavior
    use_chord_spacer: bool = False

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        logging.debug(f"running DumbAccompanistSettings __post_init__()")
        if isinstance(self.accompaniment_below, str):
            self.accompaniment_below = [self.accompaniment_below]


# (Malcolm 2023-11-14) this can be removed and replaced with DumbAccompanist2
# class DumbAccompanist:
#     def __init__(
#         self,
#         score_interface: AccompanimentInterface,
#         voices_to_accompany: t.Sequence[Voice],
#         settings: t.Optional[DumbAccompanistSettings] = None,
#     ):
#         if settings is None:
#             settings = DumbAccompanistSettings()
#         self.settings = settings
#         self._cs = SimpleSpacer()
#         self._iq = IntervalQuerier()
#         self._score_interface = score_interface
#         if len(voices_to_accompany) > 2:
#             raise ValueError()
#         self._voices_to_accompany = set(voices_to_accompany)
#         self._score_voice_attribute = "prefabs"

#         (
#             self.voice_to_keep_accompaniment_above,
#             self.voice_to_keep_accompaniment_below,
#             self.include_bass,
#         ) = self._get_voices_above_and_below()
#         self._pm: PatternMaker = PatternMaker(
#             score_interface.ts,
#             include_bass=self.include_bass,
#             pattern_changes_on_downbeats_only=self.settings.pattern_changes_on_downbeats_only,
#         )

#     def _get_voices_above_and_below(self) -> tuple[Voice | None, Voice | None, bool]:
#         # Single voices
#         if self._voices_to_accompany == {MELODY}:
#             return None, MELODY, True
#         if self._voices_to_accompany == {ALTO}:
#             return None, ALTO, True
#         if self._voices_to_accompany == {TENOR}:
#             return TENOR, None, False
#         if self._voices_to_accompany == {BASS}:
#             return BASS, None, False

#         # Voice pairs
#         if self._voices_to_accompany == {MELODY, ALTO}:
#             return None, MELODY, True
#         if self._voices_to_accompany == {MELODY, TENOR}:
#             # TODO: (Malcolm 2023-08-02) or TENOR, SOPRANO w/ bass
#             return None, TENOR, True
#         if self._voices_to_accompany == {MELODY, BASS}:
#             return BASS, MELODY, False
#         if self._voices_to_accompany == {ALTO, TENOR}:
#             # TODO: (Malcolm 2023-08-02) or ALTO, None w/ bass
#             return None, TENOR, True
#         if self._voices_to_accompany == {ALTO, BASS}:
#             # TODO: (Malcolm 2023-08-02) or divided somehow?
#             return ALTO, None, False
#         if self._voices_to_accompany == {TENOR, BASS}:
#             return TENOR, None, False
#         raise ValueError()

#     @cached_property
#     def voice_to_keep_accompaniment_below(self) -> Voice | None:
#         return self._get_voices_above_and_below()[1]

#     @cached_property
#     def voice_to_keep_accompaniment_above(self) -> Voice | None:
#         return self._get_voices_above_and_below()[0]

#     def _get_below(self) -> Pitch:
#         below_voice = self.voice_to_keep_accompaniment_below
#         if below_voice is None:
#             return self.settings.range_constraints.max_accomp_pitch
#         notes = self._score_interface.departure_attr(
#             self._score_voice_attribute, below_voice
#         )
#         return min(
#             chain(
#                 (n.pitch for n in notes),
#                 [self.settings.range_constraints.max_accomp_pitch],
#             )
#         )

#     def _get_above(self) -> Pitch:
#         above_voice = self.voice_to_keep_accompaniment_above
#         if above_voice is None:
#             return self.settings.range_constraints.min_accomp_pitch
#         notes = self._score_interface.departure_attr(
#             self._score_voice_attribute, above_voice
#         )
#         return min(
#             chain(
#                 (n.pitch for n in notes),
#                 [self.settings.range_constraints.min_accomp_pitch],
#             )
#         )

#     def _final_step(self):
#         if not self.settings.end_with_solid_chord:
#             raise NotImplementedError
#             return self.step()
#         chord = self._score_interface.departure_chord
#         assert self._score_interface.arrival_chord is None
#         below = self._get_below()
#         above = self._get_above()
#         if above >= below:
#             raise ValueError()
#         omissions = chord.get_omissions(
#             # TODO: (Malcolm 2023-07-25)
#             # existing_pitches_or_pcs=self._score_interface.get_existing_pitches(i),
#             existing_pitches_or_pcs=(),
#             # the last step should never have a suspension
#             suspensions=(),
#             iq=self._iq,
#         )
#         # pitches = [
#         #     self._score_interface.departure_pitch(voice)
#         #     for voice in self._score_interface.structural_voices
#         #     if voice not in self._voices_to_accompany
#         # ]
#         for pitches in self._cs(
#             chord.pcs,
#             omissions=omissions,
#             min_accomp_pitch=above + 1,
#             max_accomp_pitch=below - 1,
#             include_bass=self.include_bass,
#         ):
#             yield [
#                 Note(  # type:ignore
#                     pitch,
#                     chord.onset,
#                     chord.release,
#                     track=10,  # TODO: (Malcolm 2023-07-31) update
#                 )
#                 for pitch in pitches
#             ]

#     def step(
#         self, current_pitches: TwoPartResult | FourPartResult | IncrementalResult | None
#     ) -> t.Iterator[t.List[Note] | t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]:
#         assert self._pm is not None
#         assert self._score_interface.validate_state()

#         chord = self._score_interface.departure_chord
#         below = self._get_below()
#         above = self._get_above()
#         if above >= below:
#             raise ValueError()
#         # TODO: (Malcolm 2023-07-25) take account of suspensions in all voices
#         suspensions = []
#         for voice in self._score_interface.structural_voices:
#             suspension = self._score_interface.departure_suspension(voice)
#             if suspension:
#                 suspensions.append(suspension.pitch)

#         # soprano_suspension = self._score_interface.departure_suspension(
#         #     OuterVoice.MELODY
#         # )
#         # if soprano_suspension:
#         #     suspensions = (soprano_suspension.pitch,)
#         # else:
#         #     suspensions = ()
#         # TODO: (Malcolm 2023-07-25) update omissions
#         omissions = self._score_interface.departure_chord.get_omissions(
#             # LONGTERM is there anything besides structural_bass and
#             #   structural_soprano to be included in omissions?
#             existing_pitches_or_pcs=suspensions
#             if suspensions
#             else (),  # TODO: (Malcolm 2023-07-25)
#             suspensions=suspensions,
#             iq=self._iq,
#         )
#         pattern = self._pm.get_pattern(
#             chord.pcs,
#             chord.onset,
#             chord.harmony_onset,
#             chord.harmony_release,
#             pattern=self.settings.pattern,
#         )
#         spacing_constraints = self._pm.get_spacing_constraints(pattern)
#         for pitches in self._cs(
#             chord.pcs,
#             omissions=omissions,
#             min_accomp_pitch=above + 1,
#             max_accomp_pitch=below - 1,
#             include_bass=self.include_bass,
#             spacing_constraints=spacing_constraints,
#         ):
#             accompaniment_pattern = self._pm(
#                 pitches,
#                 chord.onset,
#                 release=chord.release,
#                 harmony_onset=chord.harmony_onset,
#                 harmony_release=chord.harmony_release,
#                 pattern=pattern,
#                 # TODO: (Malcolm 2023-07-25) why do we need `track` here?`
#                 track=10,  # TODO: (Malcolm 2023-07-31) update track
#                 chord_change=self._score_interface.at_chord_change(),
#             )
#             if self.settings.accompaniment_annotations is AccompAnnots.NONE:
#                 yield accompaniment_pattern
#             else:
#                 annotations = []
#                 if self.settings.accompaniment_annotations in (
#                     AccompAnnots.PATTERNS,
#                     AccompAnnots.ALL,
#                 ):
#                     assert self._pm.prev_pattern is not None
#                     annotations.append(
#                         Annotation(chord.onset, self._pm.prev_pattern.__name__)
#                     )
#                 if self.settings.accompaniment_annotations in (
#                     AccompAnnots.CHORDS,
#                     AccompAnnots.ALL,
#                 ):
#                     annotations.append(Annotation(chord.onset, chord.token))
#                 yield (accompaniment_pattern, *annotations)
#         raise AccompanimentDeadEnd("reached end of DumbAccompanist step")

#     # def init_new_piece(self, ts: t.Union[str, Meter]):
#     #     self._pm = PatternMaker(
#     #         ts,
#     #         include_bass=self.settings.include_bass,
#     #         pattern_changes_on_downbeats_only=self.settings.pattern_changes_on_downbeats_only,
#     #     )

#     def __call__(
#         self,
#         # ts: t.Optional[str] = None,
#     ) -> pd.DataFrame:
#         """If chord_data is a string, it should be a roman text file.
#         If it is a list, it should be in the same format as returned
#         by get_chords_from_rntxt

#         Keyword args:
#             ts: time signature. If chord_data is a string, this argument is
#                 ignored.

#         Returns:
#             dataframe with columns onset, text, type, pitch, release
#         """
#         # if chord_data is None and score is None:
#         #     raise ValueError("either chord_data or score must not be None")
#         # if score is None:
#         #     if isinstance(chord_data, str):
#         #         ts = get_ts_from_rntxt(chord_data)
#         #         chord_data = get_chords_from_rntxt(chord_data)
#         #     score = PrefabScore(chord_data, ts=ts)  # type:ignore

#         # assert ts is not None
#         raise NotImplementedError()

#         # self.init_new_piece(ts)
#         assert self._score_interface.empty
#         while not self._score_interface.complete:
#             result = next(self.step())
#             if self.settings.accompaniment_annotations is not AccompAnnots.NONE:
#                 raise NotImplementedError()
#                 it = iter(result)
#                 score.accompaniments.append(next(it))
#                 if self.settings.accompaniment_annotations in (
#                     AccompAnnots.ALL,
#                     AccompAnnots.PATTERNS,
#                 ):
#                     score.annotations["patterns"].append(next(it))  # type:ignore
#                 if self.settings.accompaniment_annotations in (
#                     AccompAnnots.ALL,
#                     AccompAnnots.CHORDS,
#                 ):
#                     score.annotations["chords"].append(next(it))  # type:ignore
#             else:
#                 self._score_interface.accompaniments.append(result)  # type:ignore
#         # TODO: (Malcolm 2023-07-28) return value
#         # return self._score.get_df("accompaniments")


class DumbAccompanist2(RecursiveWorker):
    def __init__(
        self,
        score: ScoreWithAccompaniments,
        voices_to_accompany: t.Sequence[Voice],
        settings: t.Optional[DumbAccompanistSettings] = None,
    ):
        super().__init__()
        if settings is None:
            settings = DumbAccompanistSettings()
        self.settings = settings
        self._cs = SimpleSpacer()
        self._iq = IntervalQuerier()
        self._score_interface = AccompanimentInterface(score)
        if len(voices_to_accompany) > 2:
            raise ValueError()
        self._voices_to_accompany = set(voices_to_accompany)
        self._score_voice_attribute = "prefabs"

        (
            self.voice_to_keep_accompaniment_above,
            self.voice_to_keep_accompaniment_below,
            self.include_bass,
        ) = self._get_voices_above_and_below()
        self._pm: PatternMaker = PatternMaker(
            score.ts,
            include_bass=self.include_bass,
            pattern_changes_on_downbeats_only=self.settings.pattern_changes_on_downbeats_only,
        )

    @cached_property
    def _voices_to_include(self) -> tuple[Voice, ...]:
        if self.voice_to_keep_accompaniment_above is None:
            lower_voice_i = 0
        else:
            lower_voice_i = [BASS, TENOR, ALTO, MELODY].index(
                self.voice_to_keep_accompaniment_above
            ) + 1
        if not self.include_bass:
            lower_voice_i = max(1, lower_voice_i)
        if self.voice_to_keep_accompaniment_below is None:
            upper_voice_i = None
        else:
            upper_voice_i = [BASS, TENOR, ALTO, MELODY].index(
                self.voice_to_keep_accompaniment_below
            )
        return (BASS, TENOR, ALTO, MELODY)[lower_voice_i:upper_voice_i]

    def _get_voices_above_and_below(self) -> tuple[Voice | None, Voice | None, bool]:
        if not self._voices_to_accompany:
            return None, None, True
        # Single voices
        if self._voices_to_accompany == {MELODY}:
            return None, MELODY, True
        if self._voices_to_accompany == {ALTO}:
            return None, ALTO, True
        if self._voices_to_accompany == {TENOR}:
            return TENOR, None, False
        if self._voices_to_accompany == {BASS}:
            return BASS, None, False

        # Voice pairs
        if self._voices_to_accompany == {MELODY, ALTO}:
            return None, ALTO, True
        if self._voices_to_accompany == {MELODY, TENOR}:
            # TODO: (Malcolm 2023-08-02) or TENOR, SOPRANO w/ bass
            return None, TENOR, True
        if self._voices_to_accompany == {MELODY, BASS}:
            return BASS, MELODY, False
        if self._voices_to_accompany == {ALTO, TENOR}:
            # TODO: (Malcolm 2023-08-02) or ALTO, None w/ bass
            return None, TENOR, True
        if self._voices_to_accompany == {ALTO, BASS}:
            # TODO: (Malcolm 2023-08-02) or divided somehow?
            return ALTO, None, False
        if self._voices_to_accompany == {TENOR, BASS}:
            return TENOR, None, False
        raise ValueError()

    def _get_below(self) -> Pitch:
        below_voice = self.voice_to_keep_accompaniment_below
        if below_voice is None:
            return self.settings.range_constraints.max_accomp_pitch
        notes = self._score_interface.departure_attr(
            self._score_voice_attribute, below_voice
        )
        return min(
            chain(
                (n.pitch for n in notes),
                [self.settings.range_constraints.max_accomp_pitch],
            )
        )

    def _get_above(self) -> Pitch:
        above_voice = self.voice_to_keep_accompaniment_above
        if above_voice is None:
            return self.settings.range_constraints.min_accomp_pitch
        notes = self._score_interface.departure_attr(
            self._score_voice_attribute, above_voice
        )
        return min(
            chain(
                (n.pitch for n in notes),
                [self.settings.range_constraints.min_accomp_pitch],
            )
        )

    def _chord_spacer_final_step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        if not self.settings.end_with_solid_chord:
            raise NotImplementedError
            return self.step()
        chord = self._score_interface.departure_chord

        below = self._get_below()
        above = self._get_above()
        if above >= below:
            raise ValueError()
        omissions = chord.get_omissions(
            # TODO: (Malcolm 2023-07-25)
            # existing_pitches_or_pcs=self._score_interface.get_existing_pitches(i),
            existing_pitches_or_pcs=(),
            # the last step should never have a suspension
            suspensions=(),
            iq=self._iq,
        )
        # pitches = [
        #     self._score_interface.departure_pitch(voice)
        #     for voice in self._score_interface.structural_voices
        #     if voice not in self._voices_to_accompany
        # ]
        # TODO: (Malcolm 2023-11-14) do we need to optionally yield Annotation here?
        for pitches in self._cs(
            chord.pcs,
            omissions=omissions,
            min_accomp_pitch=above + 1,
            max_accomp_pitch=below - 1,
            include_bass=self.include_bass,
        ):
            yield [
                Note(  # type:ignore
                    pitch,
                    chord.onset,
                    chord.release,
                    track=10,  # TODO: (Malcolm 2023-07-31) update
                )
                for pitch in pitches
            ]

    def _new_final_step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        chord = self._score_interface.departure_chord
        pitches = self._score_interface.departure_pitches(self._voices_to_include)
        yield [
            Note(  # type:ignore
                pitch,
                chord.onset,
                chord.release,
                track=10,  # TODO: (Malcolm 2023-07-31) update
            )
            for pitch in pitches
        ]

    def _final_step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        if self.settings.use_chord_spacer:
            yield from self._chord_spacer_final_step()
        else:
            yield from self._new_final_step()

    def _chord_spacer_step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        assert self._pm is not None
        assert self._score_interface.validate_state()

        if self.step_i == self.final_step_i:
            yield from self._final_step()
            return

        chord = self._score_interface.departure_chord
        below = self._get_below()
        above = self._get_above()
        if above >= below:
            raise ValueError()
        # TODO: (Malcolm 2023-07-25) take account of suspensions in all voices
        suspensions = []
        for voice in self._score_interface.structural_voices:
            suspension = self._score_interface.departure_suspension(voice)
            if suspension:
                suspensions.append(suspension.pitch)

        # soprano_suspension = self._score_interface.departure_suspension(
        #     OuterVoice.MELODY
        # )
        # if soprano_suspension:
        #     suspensions = (soprano_suspension.pitch,)
        # else:
        #     suspensions = ()
        # TODO: (Malcolm 2023-07-25) update omissions
        omissions = self._score_interface.departure_chord.get_omissions(
            # LONGTERM is there anything besides structural_bass and
            #   structural_soprano to be included in omissions?
            existing_pitches_or_pcs=suspensions
            if suspensions
            else (),  # TODO: (Malcolm 2023-07-25)
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
            include_bass=self.include_bass,
            spacing_constraints=spacing_constraints,
        ):
            accompaniment_pattern = self._pm(
                pitches,
                chord.onset,
                release=chord.release,
                harmony_onset=chord.harmony_onset,
                harmony_release=chord.harmony_release,
                pattern=pattern,
                # TODO: (Malcolm 2023-07-25) why do we need `track` here?`
                track=10,  # TODO: (Malcolm 2023-07-31) update track
                chord_change=self._score_interface.at_chord_change(),
            )
            if self.settings.accompaniment_annotations is AccompAnnots.NONE:
                yield accompaniment_pattern
            else:
                annotations = []
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.PATTERNS,
                    AccompAnnots.ALL,
                ):
                    assert self._pm.prev_pattern is not None
                    annotations.append(
                        Annotation(chord.onset, self._pm.prev_pattern.__name__)
                    )
                if self.settings.accompaniment_annotations in (
                    AccompAnnots.CHORDS,
                    AccompAnnots.ALL,
                ):
                    annotations.append(Annotation(chord.onset, chord.token))
                yield (accompaniment_pattern, *annotations)
        raise AccompanimentDeadEnd("reached end of DumbAccompanist step")

    def _new_step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        assert self._pm is not None
        assert self._score_interface.validate_state()

        if self.step_i == self.final_step_i:
            yield from self._final_step()
            return
        chord = self._score_interface.departure_chord
        pitches = self._score_interface.departure_pitches(self._voices_to_include)
        pcs = [p % 12 for p in pitches]

        pattern = self._pm.get_pattern(
            pcs,
            chord.onset,
            chord.harmony_onset,
            chord.harmony_release,
            pattern=self.settings.pattern,
        )
        accompaniment_pattern = self._pm(
            pitches,
            chord.onset,
            release=chord.release,
            harmony_onset=chord.harmony_onset,
            harmony_release=chord.harmony_release,
            pattern=pattern,
            # TODO: (Malcolm 2023-07-25) why do we need `track` here?`
            track=10,  # TODO: (Malcolm 2023-07-31) update track
            chord_change=self._score_interface.at_chord_change(),
        )
        if self.settings.accompaniment_annotations is AccompAnnots.NONE:
            yield accompaniment_pattern
        else:
            annotations = []
            if self.settings.accompaniment_annotations in (
                AccompAnnots.PATTERNS,
                AccompAnnots.ALL,
            ):
                assert self._pm.prev_pattern is not None
                annotations.append(
                    Annotation(chord.onset, self._pm.prev_pattern.__name__)
                )
            if self.settings.accompaniment_annotations in (
                AccompAnnots.CHORDS,
                AccompAnnots.ALL,
            ):
                annotations.append(Annotation(chord.onset, chord.token))
            yield (accompaniment_pattern, *annotations)

    def step(
        self,
    ) -> t.Iterator[
        t.Union[t.List[Note], t.Tuple[t.List[Note] | t.Tuple[Annotation], ...]]
    ]:
        if self.settings.use_chord_spacer:
            yield from self._chord_spacer_step()
        else:
            yield from self._new_step()

    @property
    def step_i(self) -> int:
        return self._score_interface.i

    @property
    def final_step_i(self) -> int:
        return len(self._score_interface._score.chords) - 1

    @property
    def ready(self) -> bool:
        return self._score_interface.i >= 0

    @property
    def finished(self) -> bool:
        return self.step_i > self.final_step_i

    @contextmanager
    def append_attempt(
        self,
        accompaniments: t.List[Note] | t.Tuple[t.List[Note] | t.Tuple[Annotation], ...],
    ):
        # Takes result of self.step() as argument
        with recursive_attempt(
            do_func=append_accompaniments,
            do_args=(accompaniments, self._score_interface.score),
            undo_func=pop_accompaniments,
            undo_args=(self._score_interface.score,),
        ):
            yield


def append_accompaniments(
    accompaniments: t.List[Note] | t.Tuple[t.List[Note] | t.Tuple[Annotation], ...],
    score: ScoreWithAccompaniments,
) -> tuple[int, int]:
    before_len = len(score.accompaniments)

    if isinstance(accompaniments, list):
        score.accompaniments.append(accompaniments)
    else:
        for x in accompaniments:
            if isinstance(x, Annotation):
                pass
                # TODO: (Malcolm 2023-11-14) implement
            elif isinstance(x, list):
                score.accompaniments.append(x)
            else:
                raise ValueError

    after_len = len(score.accompaniments)
    return before_len, after_len


def pop_accompaniments(score: ScoreWithAccompaniments):
    before_len = len(score.accompaniments)

    score.accompaniments.pop()

    after_len = len(score.accompaniments)
    return before_len, after_len
