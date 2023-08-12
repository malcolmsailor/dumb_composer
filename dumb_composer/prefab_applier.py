import logging
import random
import typing as t
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import accumulate
from numbers import Number

import pandas as pd

from dumb_composer.pitch_utils.intervals import (
    get_scalar_intervals_to_other_chord_factors,
    reduce_compound_interval,
)
from dumb_composer.pitch_utils.parts import note_lists_have_forbidden_parallels
from dumb_composer.pitch_utils.types import (
    TIME_TYPE,
    FourPartResult,
    InnerVoice,
    OuterVoice,
    ScalarInterval,
    SettingsBase,
    TimeStamp,
    TwoPartResult,
    Voice,
    voice_enum_to_string,
)
from dumb_composer.prefabs.prefab_pitches import (
    PrefabPitchBase,
    PrefabPitchDirectory,
    PrefabPitches,
    SingletonPitch,
)
from dumb_composer.prefabs.prefab_rhythms import (
    MissingPrefabError,
    PrefabBase,
    PrefabRhythmBase,
    PrefabRhythmDirectory,
    PrefabRhythms,
    SingletonRhythm,
)
from dumb_composer.shared_classes import Note, PrefabInterface, PrefabScore

from .pitch_utils.chords import Allow, Chord, Inflection
from .pitch_utils.scale import Scale, ScaleDict
from .utils.math_ import weighted_sample_wo_replacement
from .utils.recursion import DeadEnd

# TODO prefab "inertia" is implemented but it might be nice to make it more likely
#    to choose with greater probability from previous N prefabs

LOGGER = logging.getLogger(__name__)

# TODO: (Malcolm 2023-08-02) there should be a second type of parallel prefab
#   when the source prefab returns to its original pitch, then leaps to the next
#   pitch


@dataclass
class ParallelCandidate:
    voice: Voice
    departure_interval: ScalarInterval
    arrival_interval: ScalarInterval


class LeadinParallelCandidate(ParallelCandidate):
    pass


# TODO: (Malcolm 2023-08-02) better name?
class LeadoutParallelCandidate(ParallelCandidate):
    pass


class ParallelPrefabPitches:
    def __init__(
        self,
        src_prefab: PrefabPitchBase,
        departure_interval: ScalarInterval,
        arrival_interval: ScalarInterval,
    ):
        self.tie_to_next = src_prefab.tie_to_next
        self.alterations = src_prefab.alterations
        # TODO: (Malcolm 2023-08-02) if the prefab is a singleton we should
        #   just skip all parallel logic
        # departure_interval and arrival_interval are measured from
        # the `src` voice (with the existing prefab) to the `candidate` voice
        # (to which we are considering adding parallel motion). Therefore,
        #   - if the arrival interval is positive, the candidate voice is above the
        #       src voice
        #   - if the arrival interval is negative, the candidate voice is below the
        #       src voice
        if arrival_interval > 0:
            interval_delta = arrival_interval - departure_interval
        else:
            interval_delta = arrival_interval - departure_interval
        self.relative_degrees = [0] + [
            d + interval_delta for d in src_prefab.relative_degrees[1:]
        ]


def validate_parallel_prefab(parallel_prefab_pitches: ParallelPrefabPitches) -> bool:
    """
    # TODO: (Malcolm 2023-08-02) make these conditions parameters rather than hard-coded
    """
    relative_degrees = parallel_prefab_pitches.relative_degrees
    if not relative_degrees[0] == 0:
        return False
    forbidden_links = ([0, -6], [0, -8])
    for forbidden_link in forbidden_links:
        if relative_degrees[: len(forbidden_link)] == forbidden_link:
            return False
    conditional_links = (([0, 6], 5),)
    for link, following_degree in conditional_links:
        if relative_degrees[: len(link)] == link:
            try:
                actual_following_degree = relative_degrees[len(link)]
            except IndexError:
                return False
            if not actual_following_degree == following_degree:
                return False
            break
    return True


class PrefabDeadEnd(DeadEnd):
    pass


@dataclass
class PrefabApplierSettings(SettingsBase):
    # either "soprano", "tenor", or "bass"
    # prefab_voice: str = "soprano"
    prefab_voices: t.Sequence[str] = ("soprano", "tenor", "alto", "bass")

    # If None we sample uniformly from prefab_voices
    prefab_voices_weights: t.Sequence[float] | None = None
    n_voices_to_sample_at_each_step: t.Sequence[int] = (1,)

    # If None we sample uniformly from n_voices_to_sample_at_each_step
    n_voices_to_sample_at_each_step_weights: t.Sequence[float] | None = None

    # prefab_inertia is approximately "the probability of choosing the Nth
    #   last prefab applied to a segment of the same length". E.g., if the
    #   segment is length 2, and prefab_inertia is (0.5, 0.2), then there is
    #   about a 50% chance of choosing the last-chosen length-2 prefab, and
    #   a 20% chance of choosing the second-last-chosen length-2 prefab. The
    #   remaining probability mass is distributed to the other prefabs.
    #   The items in prefab_inertia should be in (0.0, 1.0) and should sum to
    #   at most 1.
    prefab_inertia: t.Optional[t.Tuple[float, ...]] = (0.5, 0.2)
    forbidden_parallels: t.Sequence[int] = (7, 0)

    # parallel "lead-in" prefabs are constructed as follows:
    #   if the "arrival" interval is in the parallel motion interval whitelist (i.e.,
    #   3rds or 6ths) and certain other conditions are fulfilled, then we move
    #   in parallel motion to the existing prefab, after the first note (which is
    #   the structural pitch). For example, if the existing prefab has
    #       C5 D5 C5 Bb4 | A4
    #   and the structural pitches in the other voice are
    #       E4 | F4
    #   then the parallel leadin prefab would be
    #       E4 Bb4 A4 G4 | F4
    # The "lead-in" nomenclature comes from the fact that we are "leading-in" to
    #   the following pitch
    # In practice this doesn't work very well, because there are too many other
    # conditions that need to be met. I think it might be worth trying to implement
    # a condition that the leapt-to pitch that starts the parallel leadin prefab (i.e.,
    # Bb4 in the preceding example) must either be:
    #   1. a chord tone
    #   2. immediately followed by a chord tone by step (possibly in contrary motion)
    allow_parallel_leadin_prefabs: bool = False

    # parallel "lead-out" prefabs are constructed by simply moving the existing
    # prefab in parallel motion, provided
    allow_parallel_leadout_prefabs: bool = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type:ignore
        assert self.prefab_voices
        if self.prefab_voices_weights is not None:
            assert len(self.prefab_voices_weights) == len(self.prefab_voices)
        assert self.n_voices_to_sample_at_each_step
        assert max(self.n_voices_to_sample_at_each_step) <= len(self.prefab_voices)
        if self.n_voices_to_sample_at_each_step_weights is not None:
            assert len(self.n_voices_to_sample_at_each_step_weights) == len(
                self.n_voices_to_sample_at_each_step
            )
            self.n_voices_to_sample_at_each_step_weights = tuple(
                accumulate(self.n_voices_to_sample_at_each_step_weights)
            )


class PrefabApplier:
    prefab_rhythm_dir = PrefabRhythmDirectory()
    prefab_pitch_dir = PrefabPitchDirectory()

    def __init__(
        self,
        score_interface: PrefabInterface,
        settings: t.Optional[PrefabApplierSettings] = None,
    ):
        if settings is None:
            settings = PrefabApplierSettings()
        self.settings = settings
        self._prefab_rhythm_stacks: t.DefaultDict[
            TIME_TYPE, t.List[PrefabRhythmBase]
        ] = defaultdict(list)
        self._prefab_pitch_stacks: defaultdict[
            TIME_TYPE, t.List[PrefabPitchBase]
        ] = defaultdict(list)
        # TODO: (Malcolm 2023-07-28) NB four-part composer settings has range
        # constraints in settings. Does this matter?
        self._score_interface = score_interface
        self.missing_prefabs: Counter[str] = Counter()

    def _get_parallel_prefab_candidates(
        self,
        decorated_voices: t.Sequence[Voice],
        voices_not_to_decorate: t.Sequence[Voice],
        allowed_arrival_intervals: t.Container[ScalarInterval] = frozenset({2, 5}),
        allowed_departure_intervals: t.Container[ScalarInterval] = frozenset({2, 5}),
    ) -> dict[Voice, list[ParallelCandidate]]:
        # TODO: (Malcolm 2023-08-02) would be nice to allow combinations of
        #   3 voices too (e.g., parallel 6/3 triads)
        out = defaultdict(list)
        allow_leadin = self.settings.allow_parallel_leadin_prefabs
        allow_leadout = self.settings.allow_parallel_leadout_prefabs
        if not allow_leadin and not allow_leadout:
            return out

        departure_scale = self._score_interface.departure_scale
        arrival_scale = self._score_interface.arrival_scale
        for decorated_voice in decorated_voices:
            decorated_arrival_pitch = self._score_interface.arrival_pitch(
                decorated_voice
            )
            decorated_departure_pitch = self._score_interface.departure_pitch(
                decorated_voice
            )
            for candidate_voice in voices_not_to_decorate:
                if self._score_interface.departure_is_suspension(candidate_voice):
                    # TODO: (Malcolm 2023-08-02) we may want to allow suspensions
                    #    in certain circumstances
                    continue
                if candidate_voice is OuterVoice.BASS:
                    # TODO: (Malcolm 2023-08-02) we should allow bass as long as it
                    #   doesn't leap downwards (or perhaps other constraints?)
                    continue
                candidate_arrival_pitch = self._score_interface.arrival_pitch(
                    candidate_voice
                )
                arrival_interval = arrival_scale.get_interval(
                    decorated_arrival_pitch,
                    candidate_arrival_pitch,
                )
                departure_interval = departure_scale.get_interval(
                    decorated_departure_pitch,
                    self._score_interface.departure_pitch(candidate_voice),
                )
                reduced_arrival_interval = reduce_compound_interval(
                    arrival_interval, n_steps_per_octave=len(arrival_scale)
                )

                if (
                    allow_leadin
                    and abs(reduced_arrival_interval) in allowed_arrival_intervals
                ):
                    out[decorated_voice].append(
                        LeadinParallelCandidate(
                            candidate_voice, departure_interval, arrival_interval
                        )
                    )
                reduced_departure_interval = reduce_compound_interval(
                    arrival_interval, n_steps_per_octave=len(arrival_scale)
                )
                if (
                    allow_leadout
                    and abs(reduced_departure_interval) in allowed_arrival_intervals
                ):
                    out[decorated_voice].append(
                        LeadoutParallelCandidate(
                            candidate_voice, departure_interval, arrival_interval
                        )
                    )
        return out

    def _get_parallel_leadin_candidate(
        self,
        candidate: LeadinParallelCandidate,
        src_prefab_pitch: PrefabPitchBase,
        src_prefab_rhythm: PrefabRhythmBase,
    ) -> None | list[Note]:
        departure_pitch = self._score_interface.departure_pitch(candidate.voice)
        parallel_prefab_pitches = ParallelPrefabPitches(
            src_prefab_pitch,
            candidate.departure_interval,
            candidate.arrival_interval,
        )
        if not validate_parallel_prefab(parallel_prefab_pitches):
            return None
        notes = self._make_notes_from_prefabs(
            departure_pitch, parallel_prefab_pitches, src_prefab_rhythm
        )
        return notes

    def _check_if_prefab_matches_criteria(
        self,
        src_prefab_pitch: PrefabPitchBase,
        src_prefab_rhythm: PrefabRhythmBase,
        voice: Voice,
    ):
        interval_to_next = (
            self._score_interface.structural_interval_from_departure_to_arrival(voice)
        )
        metric_strength_str = src_prefab_rhythm.metric_strength_str  # type:ignore
        departure_structural_interval = (
            self._score_interface.departure_interval_above_bass(voice)
        )
        relative_chord_factors = get_scalar_intervals_to_other_chord_factors(
            departure_structural_interval,
            self._score_interface.departure_chord.scalar_intervals_including_bass,
            len(self._score_interface.departure_scale),
        )
        is_suspension = self._score_interface.departure_is_suspension(voice)
        is_preparation = self._score_interface.departure_is_preparation(voice)
        interval_is_diatonic = self._score_interface.departure_scale.pitch_is_diatonic(
            self._score_interface.arrival_pitch(voice)
        )
        return src_prefab_pitch.matches_criteria(
            interval_to_next,
            metric_strength_str,
            relative_chord_factors,
            is_suspension,
            is_preparation,
            interval_is_diatonic,
        )

    def _get_parallel_leadout_candidate(
        self,
        candidate: LeadoutParallelCandidate,
        src_prefab_pitch: PrefabPitchBase,
        src_prefab_rhythm: PrefabRhythmBase,
    ) -> None | list[Note]:
        # if not src_prefab_pitch.returns_to_main_pitch():
        #     return None
        departure_pitch = self._score_interface.departure_pitch(candidate.voice)
        notes = self._make_notes_from_prefabs(
            departure_pitch, src_prefab_pitch, src_prefab_rhythm
        )
        return notes

    def _yield_parallel_prefabs(
        self,
        notes_so_far: dict[Voice, list[Note]],
        prefabs_so_far: dict[Voice, tuple[PrefabRhythmBase, PrefabPitchBase]],
        parallel_prefab_candidates: dict[Voice, list[ParallelCandidate]],
    ) -> t.Iterator[dict[Voice, list[Note]]]:
        srcs = random.sample(prefabs_so_far.items(), k=len(prefabs_so_far))
        for src_voice, (src_prefab_rhythm, src_prefab_pitch) in srcs:
            candidates = parallel_prefab_candidates[src_voice]
            random.shuffle(candidates)
            for candidate in candidates:
                if isinstance(candidate, LeadinParallelCandidate):
                    src_structural_interval = self._score_interface.structural_interval_from_departure_to_arrival(
                        src_voice
                    )

                    if not (
                        src_prefab_pitch.stepwise_internally()
                        and src_prefab_pitch.approaches_arrival_pitch_by_step(
                            src_structural_interval
                        )
                    ):
                        continue

                    notes = self._get_parallel_leadin_candidate(
                        candidate, src_prefab_pitch, src_prefab_rhythm
                    )
                elif isinstance(candidate, LeadoutParallelCandidate):
                    if not self._check_if_prefab_matches_criteria(
                        src_prefab_pitch, src_prefab_rhythm, candidate.voice
                    ):
                        continue
                    notes = self._get_parallel_leadout_candidate(
                        candidate, src_prefab_pitch, src_prefab_rhythm
                    )
                else:
                    raise ValueError()
                if notes is not None:
                    notes_so_far_with_parallel_voice_removed = {
                        v: n
                        for (v, n) in notes_so_far.items()
                        if v is not candidate.voice
                    }
                    if not self._has_forbidden_parallels(
                        notes, candidate.voice, notes_so_far_with_parallel_voice_removed
                    ):
                        LOGGER.debug(
                            f"{self.__class__.__name__} yielding parallel notes "
                            + " ".join(str(note) for note in notes)
                        )
                        yield {
                            candidate.voice: notes
                        } | notes_so_far_with_parallel_voice_removed

    def _make_notes_from_prefabs(
        self,
        initial_pitch: int,
        prefab_pitches: PrefabPitchBase | ParallelPrefabPitches,
        prefab_rhythms: PrefabRhythmBase,
        # TODO: (Malcolm 2023-07-28) set track
        track: int = 1,
    ) -> t.List[Note]:
        notes = []
        scale = self._score_interface.departure_scale
        # TODO if and when we allow "chromatic" suspensions (i.e.,
        #   suspensions that belong to the chord/scale of the preparation
        #   but not to the chord/scale of the suspension) scale.index can
        #   fail.
        orig_scale_degree = scale.index(initial_pitch)
        if isinstance(prefab_rhythms, SingletonRhythm):
            releases = [
                self._score_interface.departure_chord.release
                - self._score_interface.departure_chord.onset
            ]
        else:
            releases = prefab_rhythms.releases
        offset = self._score_interface.departure_chord.onset

        assert releases is not None

        for i, (rel_scale_degree, onset, release) in enumerate(
            zip(prefab_pitches.relative_degrees, prefab_rhythms.onsets, releases)
        ):
            if i in prefab_pitches.alterations:
                new_pitch = scale.get_auxiliary(
                    orig_scale_degree + rel_scale_degree,
                    prefab_pitches.alterations[i],
                )
            else:
                new_pitch = scale[orig_scale_degree + rel_scale_degree]
            if (
                i > 0
                and abs(
                    (
                        current_mod_degree := (orig_scale_degree + rel_scale_degree)
                        % len(scale)
                    )
                    - (
                        prev_mod_degree := (
                            orig_scale_degree + prefab_pitches.relative_degrees[i - 1]
                        )
                        % len(scale)
                    )
                )
                in (1, len(scale) - 1)
                and abs(new_pitch - notes[-1].pitch) > 2
            ):
                # The abs(new_pitch - notes[-1].pitch) > 2 condition is
                #   necessary because we only want to run this logic if there
                #   actually *is* an augmented second because of the case where
                #   an augmented second is immediately repeated. E.g., given
                #   pitches (G#, F, G#), if we raise the F after processing the
                #   first pair of pitches, we would raise it *again* after
                #   processing the second pair.
                aug2nds = (
                    self._score_interface.departure_chord.augmented_second_adjustments
                )
                if all(
                    degree in aug2nds
                    for degree in (prev_mod_degree, current_mod_degree)
                ):
                    if (
                        aug2nds[current_mod_degree] is Inflection.NONE
                        and aug2nds[prev_mod_degree] is Inflection.NONE
                    ):
                        pass
                    elif aug2nds[current_mod_degree] is Inflection.UP:
                        new_pitch += 1
                    elif aug2nds[current_mod_degree] is Inflection.DOWN:
                        new_pitch -= 1
                    elif aug2nds[prev_mod_degree] is Inflection.UP:
                        notes[-1].pitch += 1
                    elif aug2nds[prev_mod_degree] is Inflection.DOWN:
                        notes[-1].pitch -= 1
                    else:
                        # Inflection of both pitches must be EITHER; we adjust
                        # the previous pitch.
                        if new_pitch > notes[-1].pitch:
                            notes[-1].pitch += 1
                        else:
                            notes[-1].pitch -= 1
            notes.append(
                Note(  # type:ignore
                    new_pitch, onset + offset, release + offset, track=track
                )
            )
        if prefab_pitches.tie_to_next:
            notes[-1].tie_to_next = True
        # TODO I need to finish implementing augmented 2nds here
        # elif abs(notes[-1].pitch - next_mel_pitch) > 2 and :
        #     pass
        return notes

    @cached_property
    def decorated_voices(self) -> t.Tuple[Voice, ...]:
        out = []
        if "bass" in self.settings.prefab_voices:
            out.append(OuterVoice.BASS)
        if "tenor" in self.settings.prefab_voices:
            out.append(InnerVoice.TENOR)
        if "alto" in self.settings.prefab_voices:
            out.append(InnerVoice.ALTO)
        if (
            "melody" in self.settings.prefab_voices
            or "soprano" in self.settings.prefab_voices
        ):
            out.append(OuterVoice.MELODY)
        return tuple(out)

    def _get_prefab_weights(
        self,
        prefab_options: list[PrefabBase],
        prefab_stack_key: t.Any,
        prefab_type: t.Type,
    ) -> t.List[float]:
        """Weights will not sum to 1 in those cases where all prefab_options
        are in the first n items of the stack.

        # TODO: (Malcolm 2023-07-31) restore?
        # >>> prefab_applier = PrefabApplier()
        # >>> segment_dur = 3.0
        # >>> rhythm_options = prefab_applier.prefab_rhythm_dir(segment_dur)
        # >>> prefab_applier._get_prefab_weights(  # doctest: +SKIP
        # ...     prefab_options=rhythm_options,
        # ...     prefab_stack_key=segment_dur,
        # ...     prefab_type=PrefabRhythms,
        # ... )
        # [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        """
        if prefab_type is PrefabRhythms:
            stacks = self._prefab_rhythm_stacks
        else:
            stacks = self._prefab_pitch_stacks
        stack = stacks[prefab_stack_key]
        prefab_inertia = self.settings.prefab_inertia
        temp = {}
        remaining_mass = 1.0
        for prefab, prob in zip(reversed(stack), prefab_inertia):  # type:ignore
            if prefab in prefab_options:
                temp[prefab] = prob
                remaining_mass -= prob
        if len(prefab_options) == len(temp):
            return [temp.get(prefab) for prefab in prefab_options]  # type:ignore
        epsilon = remaining_mass / (len(prefab_options) - len(temp))
        return [temp.get(prefab, epsilon) for prefab in prefab_options]

    def _recurse(
        self,
        voices_to_decorate: t.Iterable[Voice],
        result_so_far: dict[Voice, list[Note]] | None = None,
        prefabs_so_far: dict[Voice, tuple[PrefabRhythmBase, PrefabPitchBase]]
        | None = None,
    ) -> t.Iterator[
        tuple[
            dict[Voice, list[Note]],
            dict[Voice, tuple[PrefabRhythmBase, PrefabPitchBase]],
        ]
    ]:
        if result_so_far is None:
            result_so_far = {}
        if prefabs_so_far is None:
            prefabs_so_far = {}
        decorated_voice, *remaining_voices = voices_to_decorate
        departure_pitch = self._score_interface.departure_pitch(decorated_voice)
        arrival_pitch = self._score_interface.arrival_pitch(decorated_voice)

        departure_structural_interval = (
            self._score_interface.departure_interval_above_bass(decorated_voice)
        )

        # TODO I use segment_dur as key to the prefab_stacks. But I think
        #   maybe I should use a string that combines the segment_dur with
        #   the metric weight of the start and end point.

        is_suspension = self._score_interface.departure_is_suspension(decorated_voice)
        is_preparation = self._score_interface.departure_is_preparation(decorated_voice)
        segment_dur = (
            self._score_interface.departure_chord.release
            - self._score_interface.departure_chord.onset
        )
        # TODO: (Malcolm 2023-07-31) handle missing rhythm options?
        rhythm_options = self.prefab_rhythm_dir(
            segment_dur,  # type:ignore
            # TODO metric strength
            is_suspension=is_suspension,
            is_preparation=is_preparation,
            is_resolution=self._score_interface.departure_is_resolution(
                decorated_voice
            ),
            is_after_tie=self._score_interface.departure_follows_tie(decorated_voice),
            start_with_rest=self._score_interface.get_departure_can_start_with_rest(
                decorated_voice
            ),
        )
        rhythm_weights = self._get_prefab_weights(
            rhythm_options, segment_dur, PrefabRhythms  # type:ignore
        )
        for rhythm in weighted_sample_wo_replacement(rhythm_options, rhythm_weights):
            LOGGER.debug(f"{decorated_voice=} trying rhythm {str(rhythm)}")
            self._prefab_rhythm_stacks[segment_dur].append(rhythm)
            self._score_interface.set_arrival_can_start_with_rest(
                decorated_voice,
                rhythm.allow_next_to_start_with_rest,  # type:ignore
            )
            generic_melody_pitch_interval = (
                self._score_interface.departure_scale.get_interval(
                    departure_pitch,
                    arrival_pitch,
                    scale2=self._score_interface.arrival_scale,
                )
            )
            interval_is_diatonic = (
                self._score_interface.departure_scale.pitch_is_diatonic(arrival_pitch)
            )

            relative_chord_factors = get_scalar_intervals_to_other_chord_factors(
                departure_structural_interval,
                self._score_interface.departure_chord.scalar_intervals_including_bass,
                len(self._score_interface.departure_scale),
            )
            try:
                pitch_options = self.prefab_pitch_dir(
                    generic_melody_pitch_interval,
                    rhythm.metric_strength_str,
                    decorated_voice,
                    relative_chord_factors,
                    is_suspension=is_suspension,
                    is_preparation=is_preparation,
                    interval_is_diatonic=interval_is_diatonic,
                )
            except MissingPrefabError as exc:
                LOGGER.debug(f"{exc.__class__.__name__}: {str(exc)}")
                self.missing_prefabs[str(exc)] += 1
                continue
            pitch_weights = self._get_prefab_weights(
                # I'm not completely sure what the appropriate value for
                #   "prefab_stack_key" is; trying "segment_dur". If I change
                #   it I also need to change it when appending/popping from
                #   self._prefab_pitch_stacks below
                pitch_options,  # type:ignore
                segment_dur,
                PrefabPitches,
            )
            for pitches in weighted_sample_wo_replacement(pitch_options, pitch_weights):
                LOGGER.debug(
                    f"{decorated_voice=} trying prefab pitch intervals {str(pitches.relative_degrees)}"
                )
                self._prefab_pitch_stacks[segment_dur].append(pitches)
                if pitches.tie_to_next:
                    self._score_interface.add_tie_from_departure(decorated_voice)
                notes = self._make_notes_from_prefabs(
                    departure_pitch,
                    pitches,
                    rhythm,
                )

                if not self._has_forbidden_parallels(
                    new_prefab=notes,
                    new_prefab_voice=decorated_voice,
                    existing_prefabs=result_so_far,
                ):
                    # It's not totally clear what the right way to check for forbidden
                    # parallels is. The prefabs only contain the notes within each
                    # "structural" segment, but we want to check *across* segments
                    # as well. However, when we are populating a segment with prefabs,
                    # we don't have the prefabs for the next segment yet. So we
                    # can check for parallels with the *preceding* segment, however,
                    # that seems likely to lead to either:
                    #   1. a lot of unnecessary paths down deadends (since we may have
                    #       to back up to the preceding segment if all prefabs begin
                    #       with the given pitch)
                    #   2. heavy reliance on prefabs that don't begin with the
                    #       structural pitch since where the structural pitch would
                    #       produce a forbidden parallel all alternatives that
                    #       begin with the structural pitch will fail.
                    # Therefore I think I'm going to implement the following strategy:
                    #   1. test "forward": when adding prefabs, test that they don't
                    #       create parallels when the next structural pitch is appended
                    #   2. test "backward": when adding prefabs, test that they don't
                    #       create parallels with the last pitch of the preceding
                    #       prefabs.
                    LOGGER.debug(
                        f"{self.__class__.__name__} yielding notes "
                        + " ".join(str(note) for note in notes)
                    )
                    out = {decorated_voice: notes} | result_so_far
                    prefabs_out = prefabs_so_far | {decorated_voice: (rhythm, pitches)}
                    if not remaining_voices:
                        yield out, prefabs_out
                    else:
                        yield from self._recurse(remaining_voices, out, prefabs_out)
                else:
                    LOGGER.debug("forbidden_parallels")

                if pitches.tie_to_next:
                    self._score_interface.remove_tie_from_departure(decorated_voice)
                self._prefab_pitch_stacks[segment_dur].pop()
            LOGGER.debug("no more prefab pitches for this rhythm")
            self._score_interface.unset_arrival_can_start_with_rest(decorated_voice)
            self._prefab_rhythm_stacks[segment_dur].pop()
        raise PrefabDeadEnd("reached end of Prefab _recurse()")

    def _fill_in_undecorated_voices(
        self, voices_not_to_decorate: t.Sequence[Voice]
    ) -> dict[Voice, list[Note]]:
        out = {}
        departure_chord = self._score_interface.departure_chord
        for voice in voices_not_to_decorate:
            departure_pitch = self._score_interface.departure_pitch(voice)
            out[voice] = [
                Note(departure_pitch, departure_chord.onset, departure_chord.release)
            ]
        return out

    def _get_n_voices_to_sample(self) -> int:
        n_voices_to_sample = self.settings.n_voices_to_sample_at_each_step
        if len(n_voices_to_sample) == 1:
            return n_voices_to_sample[0]
        return random.choices(
            n_voices_to_sample,
            cum_weights=self.settings.n_voices_to_sample_at_each_step_weights,
            k=1,
        )[0]

    def _sample_voices_to_decorate(self) -> tuple[list[Voice], list[Voice]]:
        """
        Takes:
        - n of voices to decorate at each step (should be probabilistic?)
        - samples from voices according to weights wo/ replacement until *n* is reached
        """
        n_voices_to_sample = self._get_n_voices_to_sample()
        voices = self.decorated_voices
        voice_weights = self.settings.prefab_voices_weights
        if voice_weights is None:
            voices_to_decorate = random.sample(voices, k=n_voices_to_sample)
        else:
            sample_iter = weighted_sample_wo_replacement(voices, voice_weights)
            voices_to_decorate = [
                v for v, _ in zip(sample_iter, range(n_voices_to_sample))
            ]
        voices_not_to_decorate = [
            voice for voice in voices if voice not in voices_to_decorate
        ]
        return voices_to_decorate, voices_not_to_decorate

    def step(self) -> t.Iterator[dict[Voice, list[Note]]]:
        assert self._score_interface.validate_state()
        voices_to_decorate, voices_not_to_decorate = self._sample_voices_to_decorate()
        undecorated_voices = self._fill_in_undecorated_voices(voices_not_to_decorate)
        for result, prefabs in self._recurse(
            voices_to_decorate, result_so_far=undecorated_voices
        ):
            parallel_prefab_candidates = self._get_parallel_prefab_candidates(
                voices_to_decorate, voices_not_to_decorate
            )
            # TODO: (Malcolm 2023-08-02) make probabilistic
            for item in self._yield_parallel_prefabs(
                result, prefabs, parallel_prefab_candidates
            ):
                yield item

            yield result
        raise PrefabDeadEnd("reached end of prefab _step()")

    def _has_forbidden_parallels(
        self,
        new_prefab: list[Note],
        new_prefab_voice: Voice,
        existing_prefabs: dict[Voice, list[Note]],
    ) -> bool:
        """
        We check:
            1. prefab realizations against one another
                todo implement
            2. prefab realizations against structural parts
        We don't check structural parts against one another (we assume that the process
        that produced the structural parts is responsible for that).
        """

        this_next_pitch = self._score_interface.arrival_pitch(new_prefab_voice)
        this_mel_interval = this_next_pitch - new_prefab[-1].pitch

        for other_voice in self._score_interface.structural_voices:
            if other_voice is new_prefab_voice:
                continue
            other_next_pitch = self._score_interface.arrival_pitch(other_voice)
            harmonic_interval = abs(this_next_pitch - other_next_pitch) % 12
            if harmonic_interval not in self.settings.forbidden_parallels:
                continue
            other_prev_pitch = self._score_interface.departure_pitch(other_voice)
            other_mel_interval = other_next_pitch - other_prev_pitch

            if other_mel_interval == this_mel_interval:
                return True

        if not existing_prefabs:
            return False

        next_note_onset = self._score_interface.arrival_time
        next_note_release = next_note_onset + TimeStamp(4.0)
        this_next_note = Note(
            this_next_pitch, onset=next_note_onset, release=next_note_release
        )
        this_prev_note = self._score_interface.last_existing_prefab_note(
            new_prefab_voice
        )
        this_note_list = (
            ([] if this_prev_note is None else [this_prev_note])
            + new_prefab
            + [this_next_note]
        )

        for other_voice, other_prefab in existing_prefabs.items():
            other_next_pitch = self._score_interface.arrival_pitch(other_voice)
            other_next_note = Note(
                other_next_pitch, onset=next_note_onset, release=next_note_release
            )
            other_prev_note = self._score_interface.last_existing_prefab_note(
                other_voice
            )
            other_note_list = (
                ([] if other_prev_note is None else [other_prev_note])
                + other_prefab
                + [other_next_note]
            )
            if note_lists_have_forbidden_parallels(this_note_list, other_note_list):
                return True

        return False

    def _final_step(self):
        # TODO eventually it would be nice to be able to decorate the last note
        #   etc.
        out = {}
        for voice in self.decorated_voices:
            pitch = self._score_interface.arrival_pitch(voice)
            out[voice] = [
                Note(
                    pitch,
                    self._score_interface.arrival_chord.onset,  # type:ignore
                    self._score_interface.arrival_chord.release,  # type:ignore
                )
            ]
        LOGGER.debug(f"{self.__class__.__name__} yielding {out=}")
        yield out

    def get_missing_prefab_str(self, reverse=True, n=None):
        if reverse:
            outer_f = reversed
        else:
            outer_f = lambda x: x
        out = []
        for key, count in outer_f(self.missing_prefabs.most_common(n=n)):
            out.append(f"{count} failures:")
            out.append(key)
        return "\n".join(out)


def append_prefabs(prefabs: dict[Voice, list[Note]], score: PrefabScore):
    for voice, notes in prefabs.items():
        if score.prefabs[voice]:
            assert notes[0].onset >= score.prefabs[voice][-1][-1].release
        score.prefabs[voice].append(notes)


def pop_prefabs(voices: t.Iterable[Voice], score: PrefabScore):
    for voice in voices:
        score.prefabs[voice].pop()
