import itertools as it
import logging
import random
import textwrap
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from statistics import mean

import pandas as pd

from dumb_composer.pitch_utils.chords import Chord, Tendency
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.scale import ScaleDict
from dumb_composer.pitch_utils.spacings import RangeConstraints, SpacingConstraints
from dumb_composer.pitch_utils.types import (
    MELODY,
    FourPartResult,
    InnerVoice,
    OuterVoice,
    Pitch,
    PitchClass,
    TimeStamp,
    Voice,
)
from dumb_composer.pitch_utils.voice_lead_chords import voice_lead_chords
from dumb_composer.shared_classes import Annotation, FourPartScore, ScoreInterface
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.suspensions import (
    Suspension,
    SuspensionCombo,
    find_suspension_release_times,
    find_suspensions,
    validate_intervals_among_suspensions,
    validate_suspension_resolution,
)
from dumb_composer.two_part_contrapuntist import (
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.math_ import softmax, weighted_sample_wo_replacement
from dumb_composer.utils.recursion import (
    DeadEnd,
    RecursionFailed,
    StructuralDeadEnd,
    UndoRecursiveStep,
    append_attempt,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


# TODO: (Malcolm 2023-07-22) I think we generally don't want 9--8 suspensions in
#   inner voices except in certain special circumstances


class InnerVoicesResult(t.TypedDict):
    tenor: Pitch
    alto: Pitch


@dataclass
class FourPartComposerSettings(
    TwoPartContrapuntistSettings,
    StructuralPartitionerSettings,
):
    max_recurse_calls: t.Optional[int] = None
    range_constraints: RangeConstraints = RangeConstraints()
    spacing_constraints: SpacingConstraints = SpacingConstraints()
    allow_upward_suspension_resolutions_inner_voices: bool | tuple[int, ...] = False

    inner_voice_suspensions_dont_cross_melody: bool = True


class FourPartWorker(TwoPartContrapuntist):
    def __init__(
        self,
        *,
        chord_data: t.Union[str, t.List[Chord]] | None = None,
        score: FourPartScore | None = None,
        settings: FourPartComposerSettings | None = FourPartComposerSettings(),
    ):
        if settings is None:
            settings = FourPartComposerSettings()
        self.settings = settings  # redundant, but done so pylance doesn't complain

        if chord_data is not None:
            score = FourPartScore(chord_data)
        else:
            assert score is not None

        get_i = lambda score: len(score.structural_bass)
        validate = (
            lambda score: len({len(pitches) for pitches in score._structural.values()})
            == 1
        )
        self._score = ScoreInterface(score, get_i=get_i, validate=validate)

        # TODO: (Malcolm 2023-08-04) don't pass score here somehow

        super().__init__(chord_data=None, score=score, settings=settings)
        logging.debug(
            textwrap.fill(f"settings: {self.settings}", subsequent_indent=" " * 4)
        )
        self._scales = ScaleDict()
        self._n_recurse_calls = 0
        self._spinner = Spinner()
        self._iq = IntervalQuerier()

        if not self.settings.allow_upward_suspension_resolutions_inner_voices:
            self._upward_suspension_resolutions_inner_voices = ()
        elif isinstance(
            self.settings.allow_upward_suspension_resolutions_inner_voices, bool
        ):
            self._upward_suspension_resolutions_inner_voices = (1,)
        else:
            self._upward_suspension_resolutions_inner_voices = (
                self.settings.allow_upward_suspension_resolutions_inner_voices
            )

        # For debugging
        self._deadend_args: dict[str, t.Any] = {}

    def _check_n_recurse_calls(self):
        if (
            self.settings.max_recurse_calls is not None
            and self._n_recurse_calls > self.settings.max_recurse_calls
        ):
            LOGGER.info(
                f"Max recursion calls {self.settings.max_recurse_calls} reached"
            )
            raise RecursionFailed(
                f"Max recursion calls {self.settings.max_recurse_calls} reached"
            )

    def _get_all_suspension_combinations(
        self, suspensions_per_inner_voice: dict[InnerVoice, list[Suspension]]
    ) -> list[SuspensionCombo]:
        tenor_pitch = self._score.prev_pitch(InnerVoice.TENOR)
        alto_pitch = self._score.prev_pitch(InnerVoice.ALTO)
        if (tenor_pitch - alto_pitch) % 12 == 0:
            return [
                {voice: suspension}
                for voice in suspensions_per_inner_voice
                for suspension in suspensions_per_inner_voice[voice]
            ]

        if InnerVoice.TENOR in suspensions_per_inner_voice:
            tenor_suspensions: list[SuspensionCombo] = [
                {InnerVoice.TENOR: suspension}
                for suspension in suspensions_per_inner_voice[InnerVoice.TENOR]
            ]
        else:
            tenor_suspensions = []

        if InnerVoice.ALTO in suspensions_per_inner_voice:
            alto_suspensions: list[SuspensionCombo] = [
                {InnerVoice.ALTO: suspension}
                for suspension in suspensions_per_inner_voice[InnerVoice.ALTO]
            ]
        else:
            alto_suspensions = []

        both_voice_suspensions = [
            tenor_suspension | alto_suspension
            for tenor_suspension, alto_suspension in it.product(
                tenor_suspensions, alto_suspensions
            )
        ]
        return tenor_suspensions + alto_suspensions + both_voice_suspensions

    def _get_valid_suspension_combinations(
        self, suspensions_per_inner_voice: dict[InnerVoice, list[Suspension]]
    ) -> list[SuspensionCombo]:
        all_suspension_combos = self._get_all_suspension_combinations(
            suspensions_per_inner_voice
        )
        if not all_suspension_combos:
            return []

        soprano_suspension = self._score.current_suspension(OuterVoice.MELODY)
        if soprano_suspension:
            suspended_pitches: tuple[Pitch, ...] = (soprano_suspension.pitch,)
        else:
            suspended_pitches = ()

        bass_suspension = self._score.current_suspension(OuterVoice.BASS)
        if bass_suspension:
            suspended_bass_pitch = bass_suspension.pitch
        else:
            suspended_bass_pitch = None

        out = []
        for combo in all_suspension_combos:
            if validate_intervals_among_suspensions(
                suspended_pitches + tuple(s.pitch for s in combo.values()),
                bass_suspension_pitch=suspended_bass_pitch,
            ):
                out.append(combo)

        return out

    def _validate_suspension_resolution(
        self, inner_voice: InnerVoice, bass_pitch: Pitch, melody_pitch: Pitch
    ) -> bool:
        resolution_pitch = self._score.current_resolution(inner_voice)
        assert resolution_pitch is not None
        if resolution_pitch < bass_pitch:
            return False
        return validate_suspension_resolution(
            resolution_pitch,
            (bass_pitch, melody_pitch),
            self._score.current_chord,
            prev_melody_pitch=self._score.prev_pitch(MELODY),
            melody_pitch=melody_pitch,
        )

    def _choose_inner_voice_suspensions(
        self,
        melody_pitch: Pitch,
        bass_pitch: Pitch,
        free_inner_voices: tuple[InnerVoice, ...],
    ) -> t.Iterator[None | tuple[SuspensionCombo, TimeStamp]]:
        assert free_inner_voices
        release_times = find_suspension_release_times(
            self._score.current_chord.onset,
            self._score.current_chord.release,
            self._score.ts,
            max_weight_diff=self.settings.max_suspension_weight_diff,
            max_suspension_dur=self.settings.max_suspension_dur,
            include_stop=self._score.next_chord is not None,
        )
        if not release_times:
            yield None
            return

        existing_suspension_pitches = []
        if self._score.current_suspension(OuterVoice.MELODY):
            existing_suspension_pitches.append(melody_pitch)
        if self._score.current_suspension(OuterVoice.BASS):
            existing_suspension_pitches.append(bass_pitch)
            suspended_bass_pitch = bass_pitch
        else:
            suspended_bass_pitch = None

        suspension_chord_pcs_to_avoid = (
            set(
                self._score.current_chord.get_pcs_that_cannot_be_added_to_existing_voicing(
                    (bass_pitch, melody_pitch), suspensions=existing_suspension_pitches
                )
            )
            | self._score.current_resolution_pcs()
        )

        # -------------------------------------------------------------------------------
        # Step 1. Find suspensions that resolve on next chord
        # -------------------------------------------------------------------------------

        # Here we depend on the fact that
        #   1. self._score.next_chord.onset is the greatest possible time that can occur
        #       in release_times
        #   2. release_times are sorted from greatest to least.
        # Thus, if self._score.next_chord.onset is in release_times, it is the first
        #   element.
        if (
            self._score.next_chord is not None
            and self._score.next_chord.onset == release_times[0]
        ):
            next_chord_release_time = self._score.next_chord.onset
            release_times = release_times[1:]

            next_foot_pc: PitchClass = self._score.next_foot_pc  # type:ignore
            next_foot_tendency = self._score.next_chord.get_pitch_tendency(next_foot_pc)
            resolution_chord_pcs_to_avoid: set[PitchClass] = (
                set() if next_foot_tendency is Tendency.NONE else {next_foot_pc}
            )
            next_chord_suspensions: dict[InnerVoice, list[Suspension]] = {}

            for inner_voice in free_inner_voices:
                prev_pitch = self._score.prev_pitch(inner_voice)

                if (
                    self.settings.inner_voice_suspensions_dont_cross_melody
                    and prev_pitch > melody_pitch
                ):
                    continue

                # Note: we don't provide `other_suspended_nonbass_pitches` here because
                #   we could then rule out valid combinations of suspensions (since
                #   these can include a 4th in combination with other intervals but not
                #   singly). Instead we validate the entire combination in
                #   self._get_valid_suspension_combinations(). However, we *do* provide
                #   `other_suspended_bass_pitch` because 4ths with a bass suspension are
                #   always ruled out.
                next_chord_suspensions[inner_voice] = find_suspensions(
                    prev_pitch,
                    preparation_chord=self._score.prev_chord,
                    suspension_chord=self._score.current_chord,
                    resolution_chord=self._score.next_chord,
                    resolve_up_by=self._upward_suspension_resolutions_inner_voices,
                    suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                    resolution_chord_pcs_to_avoid=resolution_chord_pcs_to_avoid,
                    other_suspended_bass_pitch=suspended_bass_pitch,
                    forbidden_suspension_intervals_above_bass=(
                        1,
                    ),  # TODO: (Malcolm 2023-08-02)
                )
                # TODO: (Malcolm 2023-07-21) the find_suspensions function
                #   doesn't take account of bass suspensions when calculating
                #   the interval above the bass.
        else:
            next_chord_release_time = TimeStamp(0)
            next_chord_suspensions = {}

        # -------------------------------------------------------------------------------
        # Step 2. Find suspensions that resolve during the current chord
        # -------------------------------------------------------------------------------
        if release_times:
            suspensions: dict[InnerVoice, list[Suspension]] = {}
            for inner_voice in free_inner_voices:
                prev_pitch = self._score.prev_pitch(inner_voice)
                if (
                    self.settings.inner_voice_suspensions_dont_cross_melody
                    and prev_pitch > melody_pitch
                ):
                    continue

                # Note: we don't provide `other_suspended_nonbass_pitches` here because
                #   we could then rule out valid combinations of suspensions (since
                #   these can include a 4th in combination with other intervals but not
                #   singly). Instead we validate the entire combination in
                #   self._get_valid_suspension_combinations(). However, we *do* provide
                #   `other_suspended_bass_pitch` because 4ths with a bass suspension are
                #   always ruled out.
                suspensions[inner_voice] = find_suspensions(
                    prev_pitch,
                    preparation_chord=self._score.prev_chord,
                    suspension_chord=self._score.current_chord,
                    suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                    resolve_up_by=self._upward_suspension_resolutions_inner_voices,
                    other_suspended_bass_pitch=suspended_bass_pitch,
                    forbidden_suspension_intervals_above_bass=(
                        1,
                    ),  # TODO: (Malcolm 2023-08-02)
                )
        else:
            suspensions = {}

        if not suspensions and not next_chord_suspensions:
            yield None
            return

        suspension_combos = self._get_valid_suspension_combinations(suspensions)
        next_chord_suspension_combos = self._get_valid_suspension_combinations(
            next_chord_suspensions
        )

        scores = (
            [
                mean([s.score for s in suspension_combo.values()])
                for suspension_combo in suspension_combos
            ]
            + [
                mean([s.score for s in suspension_combo.values()])
                for suspension_combo in next_chord_suspension_combos
            ]
            + [self.settings.no_suspension_score]
        )
        weights = softmax(scores)

        suspension_combos_and_release_times = (
            list(zip(suspension_combos, it.repeat(release_times)))
            + list(
                zip(next_chord_suspension_combos, it.repeat([next_chord_release_time]))
            )
            + [(None, None)]
        )

        for suspension_combo, release_times in weighted_sample_wo_replacement(
            suspension_combos_and_release_times, weights
        ):
            if suspension_combo is None:
                yield None
                continue
            assert release_times is not None and suspension_combo is not None
            if len(release_times) == 1:
                yield suspension_combo, release_times[0]
            else:
                # Note: we only try a single release time for each suspension;
                # should we be more comprehensive?
                suspension_lengths = [
                    release_time - self._score.current_chord.onset  # type:ignore
                    for release_time in release_times
                ]
                # We sample suspension releases directly proportional to the
                #   resulting suspension length.
                suspension_release = random.choices(
                    release_times, k=1, weights=suspension_lengths
                )[0]
                yield suspension_combo, suspension_release

    def _voice_lead_chords(
        self,
        melody_pitch: Pitch,
        bass_pitch: Pitch,
        tenor_resolution_pitch: Pitch | None = None,
        alto_resolution_pitch: Pitch | None = None,
    ) -> t.Iterator[InnerVoicesResult]:
        # -------------------------------------------------------------------------------
        # Calculate free inner voices
        # -------------------------------------------------------------------------------
        if tenor_resolution_pitch is not None and alto_resolution_pitch is not None:
            raise ValueError("No free inner voices")
        elif tenor_resolution_pitch is not None:
            free_inner_voices = (InnerVoice.ALTO,)
        elif alto_resolution_pitch is not None:
            free_inner_voices = (InnerVoice.TENOR,)
        else:
            free_inner_voices = (InnerVoice.TENOR, InnerVoice.ALTO)

        chord1_suspensions = {}
        chord2_suspensions = {}
        for voice in it.chain(OuterVoice, InnerVoice):
            prev_suspension = self._score.prev_suspension(voice)
            if prev_suspension is not None:
                chord1_suspensions[prev_suspension.pitch] = prev_suspension

            suspension = self._score.current_suspension(voice)
            if suspension is not None:
                chord2_suspensions[suspension.pitch] = suspension

        def _result_from_voicing(
            voicing: tuple[Pitch, ...], suspension_combo: SuspensionCombo | None = None
        ) -> InnerVoicesResult:
            if suspension_combo is not None:
                assert len(suspension_combo) == 1
                suspension_voice = list(suspension_combo)[0]
                suspension = suspension_combo[suspension_voice]
            else:
                suspension_voice = None
                suspension = None

            pitch_pair = voicing[1:-1]

            if tenor_resolution_pitch is not None:
                other_pitch_i = int(pitch_pair[0] == tenor_resolution_pitch)
                other_pitch = pitch_pair[other_pitch_i]
                assert suspension_voice is None or suspension_voice is InnerVoice.ALTO
                assert suspension is None or suspension.pitch == other_pitch
                return {"tenor": tenor_resolution_pitch, "alto": other_pitch}

            elif alto_resolution_pitch is not None:
                other_pitch_i = int(pitch_pair[0] == alto_resolution_pitch)
                other_pitch = pitch_pair[other_pitch_i]
                assert suspension_voice is None or suspension_voice is InnerVoice.TENOR
                assert suspension is None or suspension.pitch == other_pitch
                return {"tenor": other_pitch, "alto": alto_resolution_pitch}

            elif suspension is not None:
                other_pitch_i = int(pitch_pair[0] == suspension.pitch)
                other_pitch = pitch_pair[other_pitch_i]
                if suspension_voice is InnerVoice.TENOR:
                    return {"tenor": suspension.pitch, "alto": other_pitch}
                else:
                    return {"tenor": other_pitch, "alto": suspension.pitch}

            return {"tenor": pitch_pair[0], "alto": pitch_pair[1]}

        for suspensions in self._choose_inner_voice_suspensions(
            melody_pitch, bass_pitch, free_inner_voices
        ):
            if suspensions is not None:
                # Apply suspensions
                # -----------------
                suspension_combo, release_time = suspensions
                self._score.apply_suspensions(
                    suspension_combo,
                    release_time,
                    annotate=self.settings.annotate_suspensions,
                )
                if len(suspension_combo) == 2:
                    assert not alto_resolution_pitch or not tenor_resolution_pitch
                    yield {
                        "tenor": suspension_combo[InnerVoice.TENOR].pitch,
                        "alto": suspension_combo[InnerVoice.ALTO].pitch,
                    }
                else:
                    these_chord2_suspensions = chord2_suspensions | {
                        s.pitch: s for s in suspension_combo.values()
                    }
                    for voicing in voice_lead_chords(
                        self._score.prev_chord,
                        self._score.current_chord,
                        chord1_pitches=self._score.prev_pitches(),
                        chord1_suspensions=chord1_suspensions,
                        chord2_melody_pitch=melody_pitch,
                        chord2_bass_pitch=bass_pitch,
                        chord2_suspensions=these_chord2_suspensions,
                    ):
                        yield _result_from_voicing(voicing, suspension_combo)

                self._score.undo_suspensions(
                    suspension_combo, annotate=self.settings.annotate_suspensions
                )
            else:
                # Apply no suspension
                # -------------------
                for voicing in voice_lead_chords(
                    self._score.prev_chord,
                    self._score.current_chord,
                    chord1_pitches=self._score.prev_pitches(),
                    chord1_suspensions=chord1_suspensions,
                    chord2_melody_pitch=melody_pitch,
                    chord2_bass_pitch=bass_pitch,
                    chord2_suspensions=chord2_suspensions,
                ):
                    yield _result_from_voicing(voicing)

    def _resolve_inner_voice_suspensions_step(
        self,
        melody_pitch: Pitch,
        bass_pitch: Pitch,
        tenor_resolves: bool,
        alto_resolves: bool,
    ) -> t.Iterator[InnerVoicesResult]:
        LOGGER.debug(
            f"resolving inner-voice suspension: {tenor_resolves=} {alto_resolves=}"
        )
        if tenor_resolves:
            if not self._validate_suspension_resolution(
                InnerVoice.TENOR, melody_pitch=melody_pitch, bass_pitch=bass_pitch
            ):
                # If the melody doubles a resolution pc, the validation will fail.
                # It would be more efficient but rather more work to not generate the
                # melody in the first place.
                return
        if alto_resolves:
            if not self._validate_suspension_resolution(
                InnerVoice.ALTO, melody_pitch=melody_pitch, bass_pitch=bass_pitch
            ):
                return
        tenor_resolution_pitch = self._score.current_resolution(InnerVoice.TENOR)
        alto_resolution_pitch = self._score.current_resolution(InnerVoice.ALTO)
        if tenor_resolution_pitch and alto_resolution_pitch:
            yield {"tenor": tenor_resolution_pitch, "alto": alto_resolution_pitch}
        elif tenor_resolution_pitch or alto_resolution_pitch:
            yield from self._voice_lead_chords(
                melody_pitch,
                bass_pitch,
                tenor_resolution_pitch=tenor_resolution_pitch,
                alto_resolution_pitch=alto_resolution_pitch,
            )

        else:
            raise ValueError()

    def _inner_voice_step(
        self, melody_pitch, bass_pitch
    ) -> t.Iterator[InnerVoicesResult]:
        # -------------------------------------------------------------------------------
        # Condition 1: start the inner voices
        # -------------------------------------------------------------------------------
        if self._score.empty:
            for pitch_voicing in self._score.current_chord.pitch_voicings(
                min_notes=4,
                max_notes=4,
                melody_pitch=melody_pitch,
                bass_pitch=bass_pitch,
                range_constraints=self.settings.range_constraints,
                spacing_constraints=self.settings.spacing_constraints,
                shuffled=True,
            ):
                inner_voices: InnerVoicesResult = {
                    "tenor": pitch_voicing[1],
                    "alto": pitch_voicing[2],
                }
                LOGGER.debug("yielding initial {inner_voices=}")
                yield inner_voices
                return

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        tenor_resolves = self._score.current_resolution(InnerVoice.TENOR) is not None
        alto_resolves = self._score.current_resolution(InnerVoice.ALTO) is not None

        if tenor_resolves or alto_resolves:
            yield from self._resolve_inner_voice_suspensions_step(
                melody_pitch, bass_pitch, tenor_resolves, alto_resolves
            )
            return

        # -------------------------------------------------------------------------------
        # Condition 3: voice-lead the inner-voices freely
        # -------------------------------------------------------------------------------

        yield from self._voice_lead_chords(melody_pitch, bass_pitch)
        LOGGER.debug("no more inner voices")

    def _step(self) -> t.Iterator[FourPartResult]:
        assert self._score.validate_state()
        if self.settings.do_first is OuterVoice.BASS:
            for bass_pitch in self._bass_step():
                for melody_pitch in self._melody_step(bass_pitch):
                    for inner_voices in self._inner_voice_step(
                        melody_pitch, bass_pitch
                    ):
                        LOGGER.debug(f"yielding {inner_voices=}")
                        yield {
                            "bass": bass_pitch,
                            "tenor": inner_voices["tenor"],
                            "alto": inner_voices["alto"],
                            "melody": melody_pitch,
                        }
        else:
            for melody_pitch in self._melody_step():
                for bass_pitch in self._bass_step(melody_pitch):
                    for inner_voices in self._inner_voice_step(
                        melody_pitch, bass_pitch
                    ):
                        yield {
                            "bass": bass_pitch,
                            "tenor": inner_voices["tenor"],
                            "alto": inner_voices["alto"],
                            "melody": melody_pitch,
                        }
        LOGGER.debug("reached dead end")
        raise StructuralDeadEnd(
            "reached end of FourPartWorker step", **self._deadend_args
        )

    def _recurse(self) -> None:
        self._check_n_recurse_calls()

        self._spinner()
        self._n_recurse_calls += 1

        if self._score.complete:
            LOGGER.debug(f"{self.__class__.__name__}._recurse: final step")
            return

        for pitches in self._step():
            # TODO: (Malcolm 2023-08-01) update to undo suspensions
            with append_attempt(
                (
                    self._score._score.structural_bass,
                    self._score._score._structural[InnerVoice.TENOR],
                    self._score._score._structural[InnerVoice.ALTO],
                    self._score._score.structural_soprano,
                ),
                (pitches["bass"], pitches["tenor"], pitches["alto"], pitches["melody"]),
            ):
                LOGGER.debug(f"append attempt {pitches}")
                return self._recurse()

    def __call__(self):
        assert self._score.empty
        self._n_recurse_calls = 0
        self._recurse()
        return self._score

    def _debug(self):
        saved_deadends = []
        self._deadend_args = dict(save_deadends_to=saved_deadends, score=self._score)
        score = self()
        return score, saved_deadends


# TODO: (Malcolm 2023-07-21) this could probably be a function rather than a class
class FourPartComposer:
    def __init__(self, settings: FourPartComposerSettings = FourPartComposerSettings()):
        self._worker: None | FourPartWorker = None
        self.settings = settings
        self.structural_partitioner = StructuralPartitioner(settings)

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        transpose: int = 0,
    ) -> pd.DataFrame:
        print("Reading score... ", end="", flush=True)
        score = FourPartScore(chord_data, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self._worker = FourPartWorker(score=score, settings=self.settings)

        try:
            score = self._worker()
        except DeadEnd:
            raise RecursionFailed("Reached a terminal dead end")
        self._worker = None
        return score._score.get_df(
            [
                "structural_soprano",
                "structural_tenor",
                "structural_alto",
                "structural_bass",
            ]
        )

    def _debug(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        transpose: int = 0,
    ):
        print("Reading score... ", end="", flush=True)
        score = FourPartScore(chord_data, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self._worker = FourPartWorker(score=score, settings=self.settings)

        try:
            score, deadends = self._worker._debug()
        except DeadEnd:
            raise RecursionFailed("Reached a terminal dead end")
        self._worker = None
        score_df = score._score.get_df(
            [
                "structural_soprano",
                "structural_tenor",
                "structural_alto",
                "structural_bass",
            ]
        )
        deadend_dfs = [
            deadend["score"]._score.get_df(
                [
                    "structural_soprano",
                    "structural_tenor",
                    "structural_alto",
                    "structural_bass",
                ]
            )
            for deadend in deadends
        ]
        return score_df, deadend_dfs


def append_structural_pitches(pitches: FourPartResult, score: FourPartScore):
    # TODO: (Malcolm 2023-08-01) maybe we can stop using strings as keys to pitches and
    #   then avoid all this unwieldy logic
    score._structural[OuterVoice.BASS].append(pitches["bass"])
    score._structural[InnerVoice.TENOR].append(pitches["tenor"])
    score._structural[InnerVoice.ALTO].append(pitches["alto"])
    score._structural[OuterVoice.MELODY].append(pitches["melody"])


def pop_structural_pitches(score: FourPartScore):
    score._structural[OuterVoice.BASS].pop()
    score._structural[InnerVoice.TENOR].pop()
    score._structural[InnerVoice.ALTO].pop()
    score._structural[OuterVoice.MELODY].pop()
