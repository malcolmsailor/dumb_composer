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
from dumb_composer.pitch_utils.types import Pitch, PitchClass, TimeStamp
from dumb_composer.pitch_utils.voice_lead_chords import voice_lead_chords
from dumb_composer.shared_classes import Annotation, FourPartScore, InnerVoice
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.suspensions import (
    Suspension,
    find_suspension_release_times,
    find_suspensions,
    validate_intervals_among_suspensions,
    validate_suspension_resolution,
)
from dumb_composer.two_part_contrapuntist import (
    OuterVoice,
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.math_ import softmax, weighted_sample_wo_replacement
from dumb_composer.utils.recursion import DeadEnd, RecursionFailed, append_attempt

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

SuspensionCombo = dict[InnerVoice, Suspension]

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
            self._score = FourPartScore(
                chord_data, range_constraints=self.settings.range_constraints
            )
        else:
            assert score is not None
            assert score.range_constraints == self.settings.range_constraints
            self._score = score

        super().__init__(chord_data=None, score=self._score, settings=settings)
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
        self._suspension_resolutions: t.DefaultDict[
            OuterVoice | InnerVoice, t.Dict[int, int]
        ] = defaultdict(dict)

        # For debugging
        self._deadend_args: dict[str, t.Any] = {}

    def validate_state(self) -> bool:
        return (
            len(self._score.structural_melody)
            == len(self._score.structural_bass)
            == len(self._score.inner_voices[InnerVoice.TENOR])
            == len(self._score.inner_voices[InnerVoice.ALTO])
        )

    @property
    def _inner_voice_to_index(self):
        return {InnerVoice.TENOR: 0, InnerVoice.ALTO: 1}

    @property
    def prev_pitches(self) -> tuple[Pitch]:
        assert self.i > 0
        return tuple(
            sorted(
                self._score.get_existing_pitches(
                    self.i - 1, ("structural_bass", "inner_voices", "structural_melody")
                )
            )
        )

    @property
    def prev_inner_voices(self) -> tuple[Pitch, Pitch]:
        assert self.i > 0
        return (
            self._score.inner_voices[InnerVoice.TENOR][self.i - 1],
            self._score.inner_voices[InnerVoice.ALTO][self.i - 1],
        )

    def prev_pitch(self, voice: InnerVoice) -> Pitch:
        voice_i = self._inner_voice_to_index[voice]
        return self.prev_inner_voices[voice_i]

    def enumerate_prev_inner_voices(self) -> t.Iterator[tuple[InnerVoice, Pitch]]:
        return zip((InnerVoice.TENOR, InnerVoice.ALTO), self.prev_inner_voices)

    def add_suspension_resolution(self, voice: OuterVoice | InnerVoice, pitch: Pitch):
        assert not self.i + 1 in self._suspension_resolutions[voice]
        self._suspension_resolutions[voice][self.i + 1] = pitch

    def add_suspension(self, voice: OuterVoice | InnerVoice, suspension: Suspension):
        if isinstance(voice, OuterVoice):
            super().add_suspension(voice, suspension)
            return

        suspension_dict = self._score.inner_voice_suspensions[voice]
        assert self.i not in suspension_dict
        suspension_dict[self.i] = suspension

    def remove_suspension_resolution(self, voice: OuterVoice | InnerVoice):
        assert self.i + 1 in self._suspension_resolutions[voice]
        del self._suspension_resolutions[voice][self.i + 1]

    def remove_suspension(self, voice: OuterVoice | InnerVoice):
        if isinstance(voice, OuterVoice):
            super().remove_suspension(voice)
            return
        self._score.inner_voice_suspensions[voice].pop(
            self.i
        )  # raises KeyError if not present

    def annotate_suspension(self, voice: OuterVoice | InnerVoice):
        if isinstance(voice, OuterVoice):
            super().annotate_suspension(voice)
            return
        annotations_label = (
            "tenor_suspensions" if voice is InnerVoice.TENOR else "alto_suspensions"
        )
        self._score.annotations[annotations_label].append(
            Annotation(self.current_chord.onset, "S")
        )

    def remove_suspension_annotation(self, voice: OuterVoice | InnerVoice):
        annotations_label = {
            OuterVoice.BASS: "bass_suspensions",
            OuterVoice.MELODY: "melody_suspensions",
            InnerVoice.TENOR: "tenor_suspensions",
            InnerVoice.ALTO: "alto_suspensions",
        }[voice]
        popped_annotation = self._score.annotations[annotations_label].pop()
        assert popped_annotation.onset == self.current_chord.onset

    def prev_inner_voice_suspension(self, inner_voice: InnerVoice) -> Suspension | None:
        assert self.i >= 1
        return self._score.inner_voice_suspensions[inner_voice].get(self.i - 1, None)

    def inner_voice_suspension(self, inner_voice: InnerVoice) -> Suspension | None:
        return self._score.inner_voice_suspensions[inner_voice].get(self.i, None)

    def has_inner_voice_suspension(self, inner_voice: InnerVoice) -> bool:
        return self.i in self._score.inner_voice_suspensions[inner_voice]

    def has_suspension_resolution(self, voice: OuterVoice | InnerVoice) -> bool:
        return self.i in self._suspension_resolutions[voice]

    def suspension_in_any_voice(self) -> bool:
        return (
            self.has_bass_suspension
            or self.has_melody_suspension
            or self.has_inner_voice_suspension(InnerVoice.TENOR)
            or self.has_inner_voice_suspension(InnerVoice.ALTO)
        )

    def resolution_pcs(self) -> set[PitchClass]:
        out = set()
        for voice in self._suspension_resolutions:
            if self.i in self._suspension_resolutions[voice]:
                out.add(self._suspension_resolutions[voice][self.i] % 12)
        return out

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
        prev_pitch_1, prev_pitch_2 = self.prev_inner_voices
        if (prev_pitch_1 - prev_pitch_2) % 12 == 0:
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

        if self.melody_suspension:
            suspended_pitches: tuple[Pitch, ...] = (self.melody_suspension.pitch,)
        else:
            suspended_pitches = ()

        if self.bass_suspension:
            suspended_bass_pitch = self.bass_suspension.pitch
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
        resolution_pitch = self._suspension_resolutions[inner_voice][self.i]
        if resolution_pitch < bass_pitch:
            return False
        return validate_suspension_resolution(
            resolution_pitch,
            (bass_pitch, melody_pitch),
            self.current_chord,
            prev_melody_pitch=self.prev_melody_pitch,
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
            self.current_chord.onset,
            self.current_chord.release,
            self.ts,
            max_weight_diff=self.settings.max_suspension_weight_diff,
            max_suspension_dur=self.settings.max_suspension_dur,
            include_stop=self.next_chord is not None,
        )
        if not release_times:
            yield None
            return

        existing_suspension_pitches = []
        if self.melody_suspension:
            existing_suspension_pitches.append(melody_pitch)
        if self.bass_suspension:
            existing_suspension_pitches.append(bass_pitch)
            suspended_bass_pitch = bass_pitch
        else:
            suspended_bass_pitch = None

        suspension_chord_pcs_to_avoid = (
            set(
                self.current_chord.get_pcs_that_cannot_be_added_to_existing_voicing(
                    (bass_pitch, melody_pitch), suspensions=existing_suspension_pitches
                )
            )
            | self.resolution_pcs()
        )
        # Here we depend on the fact that
        #   1. self.next_chord.onset is the greatest possible time that can occur
        #       in release_times
        #   2. release_times are sorted from greatest to least.
        # Thus, if self.next_chord.onset is in release_times, it is the first
        #   element.
        if self.next_chord is not None and self.next_chord.onset == release_times[0]:
            next_chord_release_time = self.next_chord.onset
            release_times = release_times[1:]

            next_foot_pc: PitchClass = self.next_foot_pc  # type:ignore
            next_foot_tendency = self.next_chord.get_pitch_tendency(next_foot_pc)
            resolution_chord_pcs_to_avoid: set[PitchClass] = (
                set() if next_foot_tendency is Tendency.NONE else {next_foot_pc}
            )
            next_chord_suspensions: dict[InnerVoice, list[Suspension]] = {}

            for inner_voice in free_inner_voices:
                prev_pitch = self.prev_pitch(inner_voice)

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
                    preparation_chord=self.prev_chord,
                    suspension_chord=self.current_chord,
                    resolution_chord=self.next_chord,
                    resolve_up_by=self._upward_suspension_resolutions_inner_voices,
                    suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                    resolution_chord_pcs_to_avoid=resolution_chord_pcs_to_avoid,
                    other_suspended_bass_pitch=suspended_bass_pitch,
                )
                # TODO: (Malcolm 2023-07-21) the find_suspensions function
                #   doesn't take account of bass suspensions when calculating
                #   the interval above the bass.
        else:
            next_chord_release_time = TimeStamp(0)
            next_chord_suspensions = {}

        if release_times:
            suspensions: dict[InnerVoice, list[Suspension]] = {}
            for inner_voice in free_inner_voices:
                prev_pitch = self.prev_pitch(inner_voice)
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
                    preparation_chord=self.prev_chord,
                    suspension_chord=self.current_chord,
                    suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                    resolve_up_by=self._upward_suspension_resolutions_inner_voices,
                    other_suspended_bass_pitch=suspended_bass_pitch,
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
                    release_time - self.current_chord.onset  # type:ignore
                    for release_time in release_times
                ]
                # We sample suspension releases directly proportional to the
                #   resulting suspension length.
                suspension_release = random.choices(
                    release_times, k=1, weights=suspension_lengths
                )[0]
                yield suspension_combo, suspension_release

    def _apply_suspension_combo(
        self,
        suspension_combo: SuspensionCombo,
        suspension_release: TimeStamp,
    ):
        LOGGER.debug(f"applying {suspension_combo=} at {self.i=}")
        if suspension_release < self.current_chord.release:
            # If the suspension resolves during the current chord, we need to split
            #   the current chord to permit that
            self.split_current_chord_at(suspension_release)
        else:
            # Otherwise, make sure the suspension resolves at the onset of the
            #   next chord
            assert (
                self.next_chord is not None
                and suspension_release == self.next_chord.onset
            )
        suspended_pitches = []
        for inner_voice, suspension in suspension_combo.items():
            suspended_pitch = self.prev_pitch(inner_voice)
            self.add_suspension_resolution(
                inner_voice, suspended_pitch + suspension.resolves_by
            )
            self.add_suspension(inner_voice, suspension)
            if self.settings.annotate_suspensions:
                self.annotate_suspension(inner_voice)
            suspended_pitches.append(suspended_pitch)

    def _undo_suspension_combo(self, suspension_combo: SuspensionCombo):
        LOGGER.debug(f"undoing {suspension_combo=} at {self.i=}")
        for inner_voice in suspension_combo:
            self.remove_suspension_resolution(inner_voice)
            self.remove_suspension(inner_voice)
            if self.settings.annotate_suspensions:
                self.remove_suspension_annotation(inner_voice)

        if not self.suspension_in_any_voice():
            self.merge_current_chords_if_they_were_previously_split()

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

        # ---------------------------------------------------------------------------
        # melody suspensions
        # ---------------------------------------------------------------------------
        if self.prev_melody_suspension is not None:
            chord1_suspensions = {self.prev_melody_pitch: self.prev_melody_suspension}
        else:
            chord1_suspensions = {}

        if self.melody_suspension is not None:
            chord2_suspensions = {melody_pitch: self.melody_suspension}
        else:
            chord2_suspensions = {}

        # ---------------------------------------------------------------------------
        # bass suspensions
        # ---------------------------------------------------------------------------

        if self.prev_bass_suspension is not None:
            chord1_suspensions[self.prev_bass_pitch] = self.prev_bass_suspension
        if self.bass_suspension is not None:
            chord2_suspensions[bass_pitch] = self.bass_suspension

        # ---------------------------------------------------------------------------
        # inner voice suspensions
        # ---------------------------------------------------------------------------

        for inner_voice in (InnerVoice.TENOR, InnerVoice.ALTO):
            prev_suspension = self.prev_inner_voice_suspension(inner_voice)
            if prev_suspension is not None:
                chord1_suspensions[prev_suspension.pitch] = prev_suspension

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
                suspension_combo, release_time = suspensions
                LOGGER.debug(f"{suspension_combo=} at {self.i=}")
                self._apply_suspension_combo(suspension_combo, release_time)
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
                        self.prev_chord,
                        self.current_chord,
                        chord1_pitches=self.prev_pitches,
                        chord1_suspensions=chord1_suspensions,
                        chord2_melody_pitch=melody_pitch,
                        chord2_bass_pitch=bass_pitch,
                        chord2_suspensions=these_chord2_suspensions,
                    ):
                        yield _result_from_voicing(voicing, suspension_combo)
                self._undo_suspension_combo(suspension_combo)
            else:
                for voicing in voice_lead_chords(
                    self.prev_chord,
                    self.current_chord,
                    chord1_pitches=self.prev_pitches,
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
                raise DeadEnd(**self._deadend_args)
        if alto_resolves:
            if not self._validate_suspension_resolution(
                InnerVoice.ALTO, melody_pitch=melody_pitch, bass_pitch=bass_pitch
            ):
                raise DeadEnd(**self._deadend_args)

        tenor_resolution_pitch = self._suspension_resolutions[InnerVoice.TENOR].get(
            self.i, None
        )
        alto_resolution_pitch = self._suspension_resolutions[InnerVoice.ALTO].get(
            self.i, None
        )
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
        if self.empty:
            for pitch_voicing in self.current_chord.pitch_voicings(
                min_notes=4,
                max_notes=4,
                melody_pitch=melody_pitch,
                bass_pitch=bass_pitch,
                range_constraints=self.settings.range_constraints,
                spacing_constraints=self.settings.spacing_constraints,
                shuffled=True,
            ):
                LOGGER.debug(f"Pitch voicing: {pitch_voicing}")
                yield {"tenor": pitch_voicing[1], "alto": pitch_voicing[2]}
                return

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        tenor_resolves = self.has_suspension_resolution(InnerVoice.TENOR)
        alto_resolves = self.has_suspension_resolution(InnerVoice.ALTO)

        if tenor_resolves or alto_resolves:
            yield from self._resolve_inner_voice_suspensions_step(
                melody_pitch, bass_pitch, tenor_resolves, alto_resolves
            )
            return

        # -------------------------------------------------------------------------------
        # Condition 3: voice-lead the inner-voices freely
        # -------------------------------------------------------------------------------

        yield from self._voice_lead_chords(melody_pitch, bass_pitch)

    def _step(self):
        self.validate_state()
        if self.settings.do_first is OuterVoice.BASS:
            for bass_pitch in self._bass_step():
                for melody_pitch in self._melody_step(bass_pitch):
                    for inner_voices in self._inner_voice_step(
                        melody_pitch, bass_pitch
                    ):
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
        raise DeadEnd(**self._deadend_args)

    def _recurse(self) -> None:
        self._check_n_recurse_calls()

        self._spinner()
        self._n_recurse_calls += 1

        if self.complete:
            LOGGER.debug(f"{self.__class__.__name__}._recurse: final step")
            return

        for pitches in self._step():
            with append_attempt(
                (
                    self._score.structural_bass,
                    self._score.inner_voices[InnerVoice.TENOR],
                    self._score.inner_voices[InnerVoice.ALTO],
                    self._score.structural_melody,
                ),
                (pitches["bass"], pitches["tenor"], pitches["alto"], pitches["melody"]),
            ):
                LOGGER.debug(f"append attempt {pitches}")
                return self._recurse()

    def __call__(self):
        assert self.empty
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
        range_constraints: RangeConstraints = RangeConstraints(),
        transpose: int = 0,
    ) -> pd.DataFrame:
        print("Reading score... ", end="", flush=True)
        score = FourPartScore(chord_data, range_constraints, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self._worker = FourPartWorker(score=score, settings=self.settings)

        try:
            score = self._worker()
        except DeadEnd:
            raise RecursionFailed("Reached a terminal dead end")
        self._worker = None
        return score.get_df(["structural_melody", "inner_voices", "structural_bass"])

    def _debug(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        range_constraints: RangeConstraints = RangeConstraints(),
        transpose: int = 0,
    ):
        print("Reading score... ", end="", flush=True)
        score = FourPartScore(chord_data, range_constraints, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self._worker = FourPartWorker(score=score, settings=self.settings)

        try:
            score, deadends = self._worker._debug()
        except DeadEnd:
            raise RecursionFailed("Reached a terminal dead end")
        self._worker = None
        score_df = score.get_df(
            ["structural_melody", "inner_voices", "structural_bass"]
        )
        deadend_dfs = [
            deadend["score"].get_df(
                ["structural_melody", "inner_voices", "structural_bass"]
            )
            for deadend in deadends
        ]
        return score_df, deadend_dfs
