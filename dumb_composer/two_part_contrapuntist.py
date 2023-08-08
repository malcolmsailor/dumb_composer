import logging
import random
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from itertools import repeat
from numbers import Number

import pandas as pd

from dumb_composer.from_ml_out import get_chord_df
from dumb_composer.pitch_utils.chords import Chord, Tendency, is_same_harmony
from dumb_composer.pitch_utils.interval_chooser import (
    IntervalChooser,
    IntervalChooserSettings,
)
from dumb_composer.pitch_utils.intervals import (
    get_forbidden_intervals,
    interval_finder,
    is_direct_interval,
)
from dumb_composer.pitch_utils.pcs import PitchClass
from dumb_composer.pitch_utils.put_in_range import get_all_in_range
from dumb_composer.pitch_utils.spacings import RangeConstraints, SpacingConstraints
from dumb_composer.pitch_utils.types import (
    BASS,
    MELODY,
    ChromaticInterval,
    Pitch,
    TimeStamp,
    TwoPartResult,
    Weight,
)
from dumb_composer.shared_classes import (
    Annotation,
    OuterVoice,
    Score,
    ScoreInterface,
    _ScoreBase,
)
from dumb_composer.suspensions import (
    Suspension,
    find_bass_suspension,
    find_suspension_release_times,
    find_suspensions,
)
from dumb_composer.time import Meter
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf
from dumb_composer.utils.math_ import softmax, weighted_sample_wo_replacement
from dumb_composer.utils.recursion import DeadEnd, StructuralDeadEnd, UndoRecursiveStep

LOGGER = logging.getLogger(__name__)


@dataclass
class TwoPartContrapuntistSettings(IntervalChooserSettings):
    forbidden_parallels: t.Sequence[int] = (7, 0)
    forbidden_antiparallels: t.Sequence[int] = (0,)
    unpreferred_direct_intervals: t.Sequence[int] = (7, 0)
    max_interval: int = 12
    max_suspension_weight_diff: int = 1
    max_suspension_dur: t.Union[Number, str] = "bar"
    # allow_upward_suspensions_melody can be a bool or a tuple of allowed intervals.
    # If a bool and True, the only allowed suspension is by semitone.
    allow_upward_suspensions_melody: t.Union[bool, t.Tuple[int]] = False
    allow_upward_suspensions_bass: t.Union[bool, t.Tuple[int]] = False
    annotate_suspensions: bool = False  # TODO: (Malcolm 2023-07-25) restore
    # when choosing whether to insert a suspension, we put the "score" of each
    #   suspension (so far, by default 1.0) into a softmax together with the
    #   following "no_suspension_score".
    # To ensure that suspensions will be used wherever possible,
    #   `no_suspension_score` can be set to a large negative number (which
    #   will become zero after the softmax) or even float("-inf").
    no_suspension_score: float = 2.0
    allow_avoid_intervals: bool = False
    allow_steps_outside_of_range: bool = True
    range_constraints: RangeConstraints = RangeConstraints()
    spacing_constraints: SpacingConstraints = SpacingConstraints()
    expected_total_number_of_voices: int = 4
    do_first: OuterVoice = OuterVoice.BASS
    weight_harmonic_intervals: bool = True

    # Consider the following chords:
    #   F: V I ii6
    # On the second chord, we *could* suspend the bass, allowing it to resolve
    # on the third chord. This seems not out of the question but a bit irregular
    # to me. The following parameter prohibits this sort of situation, but
    # it can be disabled by setting to False.
    only_allow_bass_suspensions_if_eligible_resolution_in_current_chord: bool = True

    tendency_decay_per_measure: float = 0.75

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type:ignore


class TwoPartContrapuntist:
    def __init__(
        self,
        *,
        chord_data: t.Union[str, t.List[Chord]] | None = None,
        score: _ScoreBase | None = None,
        settings: TwoPartContrapuntistSettings | None = None,
    ):
        assert (chord_data is not None and score is None) or (
            score is not None and chord_data is None
        )

        if settings is None:
            settings = TwoPartContrapuntistSettings()
        self.settings = settings
        # TODO the size of lambda parameter for DeprecatedIntervalChooser should depend on
        #   how long the chord is. If a chord lasts for a whole note it can move
        #   by virtually any amount. If a chord lasts for an eighth note it
        #   should move by a relatively small amount.
        # TODO: (Malcolm 2023-07-17) and the above in turn should be influenced by the
        #   expected density of ornamentation. If we're embellishing in 16ths then
        #   each note should be free to move relatively widely.
        # TODO: (Malcolm 2023-08-08) the above comments may be somewhat (completely?)
        #   out of date
        self._interval_chooser = IntervalChooser(settings)
        self._lingering_tendencies: t.List[t.Dict[Pitch, Weight]] = []
        self._split_chords: t.List[int] = []

        if not self.settings.allow_upward_suspensions_melody:
            self._upward_suspension_resolutions_melody = ()
        elif isinstance(self.settings.allow_upward_suspensions_melody, bool):
            self._upward_suspension_resolutions_melody = (1,)
        else:
            self._upward_suspension_resolutions_melody = (
                self.settings.allow_upward_suspensions_melody
            )

        if not self.settings.allow_upward_suspensions_bass:
            self._upward_suspension_resolutions_bass = ()
        elif isinstance(self.settings.allow_upward_suspensions_bass, bool):
            self._upward_suspension_resolutions_bass = (1,)
        else:
            self._upward_suspension_resolutions_bass = (
                self.settings.allow_upward_suspensions_bass
            )

        if chord_data is not None:
            score = Score(chord_data)
        else:
            assert score is not None
            # assert score.range_constraints == self.settings.range_constraints

        get_i = lambda score: len(score.structural_bass)
        validate = (
            lambda score: len({len(pitches) for pitches in score._structural.values()})
            == 1
        )
        self._score = ScoreInterface(score, get_i=get_i, validate=validate)

        # For debugging
        self._deadend_args: dict[str, t.Any] = {}

    def from_ml_out(
        self,
        ml_out: t.Sequence[str],
        ts: str,
        tonic_pc: int,
        *args,
        bass_range: t.Optional[t.Tuple[int, int]] = None,
        mel_range: t.Optional[t.Tuple[int, int]] = None,
        initial_mel_chord_factor: t.Optional[int] = None,
        relative_key_annotations=True,
        **kwargs,
    ):
        chord_data = get_chord_df(
            ml_out,
            ts,
            tonic_pc,
            *args,
            relative_key_annotations=relative_key_annotations,
            **kwargs,
        )
        # TODO: (Malcolm) fix
        raise NotImplementedError
        return self(chord_data, ts, bass_range, mel_range, initial_mel_chord_factor)

    # -----------------------------------------------------------------------------------
    # Composing logic
    # -----------------------------------------------------------------------------------

    def _choose_suspension(
        self, suspension_chord_pcs_to_avoid: t.Container[PitchClass]
    ) -> t.Iterator[tuple[Suspension, TimeStamp] | None]:
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

        # MELODY ONLY
        bass_suspension = self._score.current_suspension(OuterVoice.BASS)
        if bass_suspension is not None:
            bass_suspension_pitch = bass_suspension.pitch
        else:
            bass_suspension_pitch = None

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

            # MELODY ONLY
            next_foot_pc: PitchClass = self._score.next_foot_pc  # type:ignore
            next_foot_tendency = self._score.next_chord.get_pitch_tendency(next_foot_pc)
            resolution_chord_pcs_to_avoid: set[PitchClass] = (
                set() if next_foot_tendency is Tendency.NONE else {next_foot_pc}
            )
            next_chord_suspensions = find_suspensions(
                self._score.prev_pitch(MELODY),
                preparation_chord=self._score.prev_chord,
                suspension_chord=self._score.current_chord,
                resolution_chord=self._score.next_chord,
                resolve_up_by=self._upward_suspension_resolutions_melody,
                suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                resolution_chord_pcs_to_avoid=resolution_chord_pcs_to_avoid,
                other_suspended_bass_pitch=bass_suspension_pitch,
            )
        else:
            next_chord_release_time = TimeStamp(0)
            next_chord_suspensions = []

        if release_times:
            # MELODY_ONLY
            suspensions = find_suspensions(
                self._score.prev_pitch(MELODY),
                preparation_chord=self._score.prev_chord,
                suspension_chord=self._score.current_chord,
                suspension_chord_pcs_to_avoid=suspension_chord_pcs_to_avoid,
                resolve_up_by=self._upward_suspension_resolutions_melody,
                other_suspended_bass_pitch=bass_suspension_pitch,
            )
        else:
            suspensions = []

        if not suspensions and not next_chord_suspensions:
            yield None
            return

        scores = (
            [s.score for s in suspensions]
            + [s.score for s in next_chord_suspensions]
            + [self.settings.no_suspension_score]
        )
        weights = softmax(scores)

        suspensions_and_release_times = (
            list(zip(suspensions, repeat(release_times)))
            + list(zip(next_chord_suspensions, repeat([next_chord_release_time])))
            + [(None, None)]
        )
        for suspension, release_times in weighted_sample_wo_replacement(
            suspensions_and_release_times, weights
        ):
            if suspension is None:
                yield None
            else:
                assert release_times is not None
                if len(release_times) == 1:
                    yield suspension, release_times[0]
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
                    yield suspension, suspension_release

    def _choose_suspension_bass(
        self,
    ) -> t.Iterator[tuple[Suspension, TimeStamp] | None]:
        # BASS ONLY
        if self._score.current_chord.foot == self._score.prev_foot_pc:
            # if the current chord bass is the same as the previous bass, return
            # immediately
            yield None
            return

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

        soprano_suspension = self._score.current_suspension(OuterVoice.MELODY)
        if soprano_suspension is not None:
            other_suspension_pitches = (soprano_suspension.pitch,)
        else:
            other_suspension_pitches = ()

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

            # BASS ONLY
            next_chord_suspensions = find_bass_suspension(
                src_pitch=self._score.prev_pitch(BASS),
                preparation_chord=self._score.prev_chord,
                suspension_chord=self._score.current_chord,
                resolution_chord=self._score.next_chord,
                resolve_up_by=self._upward_suspension_resolutions_bass,
                other_suspended_pitches=other_suspension_pitches,
            )
        else:
            next_chord_release_time = TimeStamp(0)
            next_chord_suspensions = []

        if release_times:
            # BASS ONLY
            suspensions = find_bass_suspension(
                src_pitch=self._score.prev_pitch(BASS),
                preparation_chord=self._score.prev_chord,
                suspension_chord=self._score.current_chord,
                resolve_up_by=self._upward_suspension_resolutions_bass,
                other_suspended_pitches=other_suspension_pitches,
            )
        else:
            suspensions = []

        if not suspensions and not next_chord_suspensions:
            yield None
            return

        scores = (
            [s.score for s in suspensions]
            + [s.score for s in next_chord_suspensions]
            + [self.settings.no_suspension_score]
        )
        weights = softmax(scores)

        suspensions_and_release_times = (
            list(zip(suspensions, repeat(release_times)))
            + list(zip(next_chord_suspensions, repeat([next_chord_release_time])))
            + [(None, None)]
        )

        for suspension, release_times in weighted_sample_wo_replacement(
            suspensions_and_release_times, weights
        ):
            if suspension is None:
                yield None
            else:
                assert release_times is not None
                if len(release_times) == 1:
                    yield suspension, release_times[0]
                else:
                    # Note: we only try a single release time for each suspension;
                    # should we be more comprehensive?
                    suspension_lengths = [
                        release_time - self._score.current_chord.onset
                        for release_time in release_times
                    ]
                    # We sample suspension releases directly proportional to the
                    #   resulting suspension length.
                    suspension_release = random.choices(
                        release_times, k=1, weights=suspension_lengths
                    )[0]
                    yield suspension, suspension_release

    def _get_tendency(
        self, intervals: t.List[int], bass_has_tendency: bool
    ) -> t.Iterable[int]:
        tendency = self._score.prev_chord.get_pitch_tendency(
            self._score.prev_pitch(MELODY)
        )
        if (
            tendency is Tendency.NONE
            or self._score.prev_chord == self._score.current_chord
        ):
            return
        if tendency is Tendency.UP:
            steps = (1, 2)
        else:
            steps = (-1, -2)
        for step in steps:
            if step in intervals:
                candidate_pitch = self._score.prev_pitch(MELODY) + step
                if bass_has_tendency and (
                    candidate_pitch % 12 == self._score.current_foot_pc
                ):
                    return
                yield candidate_pitch
                intervals.remove(step)
                return
        LOGGER.debug(
            f"resolving tendency-tone with pitch {self._score.prev_pitch(MELODY)} "
            f"would exceed range"
        )

    def _get_boundary_pitches(
        self, voice: OuterVoice, pitch_in_other_voice: Pitch | None = None
    ) -> t.Tuple[Pitch, Pitch]:
        if voice is OuterVoice.BASS:
            min_pitch = self.settings.range_constraints.min_bass_pitch
            max_pitch = self.settings.range_constraints.max_bass_pitch

            if pitch_in_other_voice is not None:
                min_pitch = max(
                    min_pitch,
                    pitch_in_other_voice
                    - self.settings.spacing_constraints.max_total_interval,
                )
                max_pitch = min(
                    max_pitch,
                    pitch_in_other_voice
                    - self.settings.spacing_constraints.min_total_interval,
                )

        elif voice is OuterVoice.MELODY:
            min_pitch = self.settings.range_constraints.min_melody_pitch
            max_pitch = self.settings.range_constraints.max_melody_pitch
            if pitch_in_other_voice is not None:
                min_pitch = max(
                    min_pitch,
                    pitch_in_other_voice
                    + self.settings.spacing_constraints.min_total_interval,
                )
                max_pitch = min(
                    max_pitch,
                    pitch_in_other_voice
                    + self.settings.spacing_constraints.max_total_interval,
                )

        else:
            raise ValueError()

        return min_pitch, max_pitch

    def _get_first_melody_pitch(self, bass_pitch: Pitch | None) -> t.Iterator[Pitch]:
        eligible_pcs = (
            self._score.current_chord.get_pcs_that_can_be_added_to_existing_voicing(
                (self._score.current_foot_pc,)
            )
        )

        min_pitch, max_pitch = self._get_boundary_pitches(
            OuterVoice.MELODY, pitch_in_other_voice=bass_pitch
        )
        yield from get_all_in_range(
            eligible_pcs, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _get_first_bass_pitch(self, melody_pitch: Pitch | None) -> t.Iterator[Pitch]:
        min_pitch, max_pitch = self._get_boundary_pitches(
            OuterVoice.BASS, pitch_in_other_voice=melody_pitch
        )

        yield from get_all_in_range(
            self._score.current_foot_pc, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _choose_intervals(
        self,
        src_other_pitch: Pitch,
        dst_other_pitch: Pitch | None,
        src_pitch: Pitch,
        dont_double_other_pc: bool,
        intervals: t.List[ChromaticInterval],
        voice_to_choose_for: OuterVoice,
        notional_other_pc: PitchClass | None = None,
    ):
        """
        Sometimes the "notional" other pitch-class is different from the "actual" other
        pitch-class (e.g., when there is a suspension). Then we can provide the
        `notional_other_pc` to avoid doubling that pc (rather than the actual other pc).
        """
        avoid_intervals = []
        avoid_harmonic_intervals = []
        direct_intervals = []
        direct_harmonic_intervals = []

        custom_weights = None

        if dst_other_pitch is None:
            harmonic_intervals = None
        else:
            harmonic_intervals = [
                src_pitch + interval - dst_other_pitch for interval in intervals
            ]

        while intervals:
            interval_i = self._interval_chooser.get_interval_indices(
                intervals,
                harmonic_intervals=harmonic_intervals,
                custom_weights=custom_weights,
                n=1,
            )[0]
            interval = intervals[interval_i]
            candidate_pitch = src_pitch + interval

            if dst_other_pitch is not None:
                assert harmonic_intervals is not None
                harmonic_interval = harmonic_intervals[interval_i]
                # Check for avoid intervals

                if dont_double_other_pc and (candidate_pitch % 12) == (
                    dst_other_pitch % 12
                    if notional_other_pc is None
                    else notional_other_pc
                ):
                    # Don't double tendency tones
                    intervals.pop(interval_i)
                    harmonic_intervals.pop(interval_i)
                    avoid_intervals.append(interval)
                    avoid_harmonic_intervals.append(harmonic_interval)
                    continue

                # Check for direct intervals
                if voice_to_choose_for is OuterVoice.BASS:
                    pitch_args = (
                        src_pitch,
                        candidate_pitch,
                        src_other_pitch,
                        dst_other_pitch,
                    )
                else:
                    pitch_args = (
                        src_other_pitch,
                        dst_other_pitch,
                        src_pitch,
                        candidate_pitch,
                    )
                if is_direct_interval(
                    *pitch_args, self.settings.unpreferred_direct_intervals
                ):
                    # We move unpreferred direct intervals into another list and only
                    # consider them later if we need to
                    intervals.pop(interval_i)
                    harmonic_intervals.pop(interval_i)
                    direct_intervals.append(interval)
                    direct_harmonic_intervals.append(harmonic_interval)
                    continue

            # Try interval

            yield candidate_pitch
            intervals.pop(interval_i)
            if harmonic_intervals is not None:
                harmonic_intervals.pop(interval_i)

        # -------------------------------------------------------------------------------
        # Try avoid intervals
        # -------------------------------------------------------------------------------
        if self.settings.allow_avoid_intervals:
            while avoid_intervals:
                interval_i = self._interval_chooser.get_interval_indices(
                    avoid_intervals, avoid_harmonic_intervals, n=1
                )[0]
                interval = avoid_intervals.pop(interval_i)
                LOGGER.warning(f"must use avoid interval {interval}")
                yield src_pitch + interval

        # -------------------------------------------------------------------------------
        # Try direct intervals
        # -------------------------------------------------------------------------------
        while direct_intervals:
            interval_i = self._interval_chooser.get_interval_indices(
                direct_intervals, direct_harmonic_intervals, n=1
            )[0]
            interval = direct_intervals.pop(interval_i)
            LOGGER.warning(f"must use direct interval {interval}")
            yield src_pitch + interval

    def _get_melody_pitch(self, current_bass_pitch: Pitch | None) -> t.Iterator[Pitch]:
        foot_tendency = self._score.current_chord.get_pitch_tendency(
            self._score.current_foot_pc
        )
        dont_double_bass_pc = foot_tendency is not Tendency.NONE

        # "no suspension" is understood as `suspension is None`
        for suspension in self._choose_suspension(
            suspension_chord_pcs_to_avoid={self._score.current_foot_pc}
            if dont_double_bass_pc
            else set()
        ):
            self._lingering_tendencies.append({})
            if suspension is not None:
                yield self._score.apply_suspension(
                    *suspension,
                    voice=OuterVoice.MELODY,
                    annotate=self.settings.annotate_suspensions,
                )
                # we only continue execution if the recursive calls fail
                #   somewhere further down the stack, in which case we need to
                #   undo the suspension.
                self._score.undo_suspension(
                    voice=OuterVoice.MELODY,
                    annotate=self.settings.annotate_suspensions,
                )
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if current_bass_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        self._score.prev_pitch(MELODY),
                        [(self._score.prev_pitch(BASS), current_bass_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.MELODY, pitch_in_other_voice=current_bass_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    self._score.prev_pitch(MELODY),
                    self._score.current_chord.pcs,
                    min_pitch,
                    max_pitch,
                    max_interval=self.settings.max_interval,
                    forbidden_intervals=forbidden_intervals,
                    allow_steps_outside_of_range=self.settings.allow_steps_outside_of_range,
                )

                # TODO for now tendency tones are only considered if there is
                #   no suspension.

                # First, try to proceed according to the tendency of the previous
                # pitch
                for pitch in self._get_tendency(intervals, dont_double_bass_pc):
                    # self._get_tendency removes intervals
                    LOGGER.debug(
                        f"{self.__class__.__name__} using tendency resolution pitch {pitch}"
                    )
                    yield pitch
                    # self._lingering_tendencies[i + 1][
                    #     pitch
                    # ] = self.settings.tendency_decay_per_measure

                # If the previous pitch does not have a tendency, or proceeding
                # according to the tendency doesn't work, try the other intervals
                yield from self._choose_intervals(
                    self._score.prev_pitch(BASS),
                    current_bass_pitch,
                    self._score.prev_pitch(MELODY),
                    dont_double_bass_pc,
                    intervals,
                    voice_to_choose_for=OuterVoice.MELODY,
                    notional_other_pc=self._score.current_foot_pc,
                )
            # self._lingering_tendencies.pop()

    def _get_bass_pitch(self, current_melody_pitch: Pitch | None) -> t.Iterator[Pitch]:
        # I'm not sure whether it's strictly necessary to calculate whether we can
        # double the melody pc, since the bass foot pc is determined and the melody
        # should not double it when not appropriate to do so.
        if current_melody_pitch is None:
            dont_double_melody_pc = False
        elif self._score.current_suspension(OuterVoice.MELODY):
            # Don't double a suspension
            dont_double_melody_pc = True
        else:
            # Don't double the pc if it is a tendency tone
            dont_double_melody_pc = (
                self._score.current_chord.get_pitch_tendency(current_melody_pitch)
                is not Tendency.NONE
            )

        for suspension in self._choose_suspension_bass():
            if suspension is not None:
                yield self._score.apply_suspension(
                    *suspension,
                    voice=OuterVoice.BASS,
                    annotate=self.settings.annotate_suspensions,
                )
                # Note: since we yield above (rather than making a recursive call), we
                #   need to be careful to undo the suspension if the recursive calls
                #   fail.
                self._score.undo_suspension(
                    voice=OuterVoice.BASS,
                    annotate=self.settings.annotate_suspensions,
                )
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if current_melody_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        self._score.prev_pitch(BASS),
                        [(self._score.prev_pitch(MELODY), current_melody_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.BASS, pitch_in_other_voice=current_melody_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    self._score.prev_pitch(BASS),
                    (self._score.current_chord.foot,),
                    min_pitch,
                    max_pitch,
                    max_interval=self.settings.max_interval,
                    forbidden_intervals=forbidden_intervals,
                    allow_steps_outside_of_range=self.settings.allow_steps_outside_of_range,
                )

                # Compared with the melody implementation, we don't worry about tendency
                # tones since the pitch-class is determined by the chord symbol in any
                # case

                yield from self._choose_intervals(
                    self._score.prev_pitch(MELODY),
                    current_melody_pitch,
                    self._score.prev_pitch(BASS),
                    dont_double_melody_pc,
                    intervals,
                    # TODO: (Malcolm 2023-08-04) this was MELODY, shouldn't it be BASS?
                    voice_to_choose_for=OuterVoice.BASS,
                )

    def _melody_step(self, bass_pitch: Pitch | None = None) -> t.Iterator[Pitch]:
        # -------------------------------------------------------------------------------
        # Condition 1: start the melody
        # -------------------------------------------------------------------------------

        if self._score.empty:
            # generate the first note
            for pitch in self._get_first_melody_pitch(bass_pitch):
                LOGGER.debug(
                    f"{self.__class__.__name__} yielding first melody pitch {pitch}"
                )
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------
        elif (pitch := self._score.current_resolution(OuterVoice.MELODY)) is not None:
            LOGGER.debug(
                f"{self.__class__.__name__} yielding melody suspension resolution pitch {pitch}"
            )
            yield pitch

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            for pitch in self._get_melody_pitch(bass_pitch):
                LOGGER.debug(f"{self.__class__.__name__} yielding melody {pitch=}")
                yield pitch

    def _bass_step(self, melody_pitch: Pitch | None = None) -> t.Iterator[Pitch]:
        # -------------------------------------------------------------------------------
        # Condition 1: start the bass
        # -------------------------------------------------------------------------------
        if self._score.empty:
            for pitch in self._get_first_bass_pitch(melody_pitch):
                LOGGER.debug(
                    f"{self.__class__.__name__} yielding first bass pitch {pitch}"
                )
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        elif (pitch := self._score.current_resolution(OuterVoice.BASS)) is not None:
            LOGGER.debug(
                f"{self.__class__.__name__} yielding bass suspension resolution pitch {pitch}"
            )
            yield pitch

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            for pitch in self._get_bass_pitch(melody_pitch):
                LOGGER.debug(f"{self.__class__.__name__} yielding bass {pitch=}")
                yield pitch

    def _step(self) -> t.Iterator[TwoPartResult]:
        assert self._score.validate_state()
        if self.settings.do_first is OuterVoice.BASS:
            for bass_pitch in self._bass_step():
                for melody_pitch in self._melody_step(bass_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}
        else:
            for melody_pitch in self._melody_step():
                for bass_pitch in self._bass_step(melody_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}

        LOGGER.debug("reached dead end")
        raise StructuralDeadEnd(
            "reached end of TwoPartContrapuntist step", **self._deadend_args
        )

    def __call__(self):
        assert self._score.empty
        while not self._score.complete:
            pitches = next(self._step())
            # TODO: (Malcolm 2023-08-04) better variable names
            self._score._score.structural_soprano.append(pitches["melody"])
            self._score._score.structural_bass.append(pitches["bass"])
        return self._score

    def get_mididf_from_score(self, score: _ScoreBase):
        out_dict = {
            "onset": [chord.onset for chord in score.chords],
            "release": [chord.release for chord in score.chords],
            "bass": score.structural_bass,
            "melody": score.structural_soprano,
        }
        out_df = pd.DataFrame(out_dict)
        return homodf_to_mididf(out_df)


def append_structural_pitches(pitches: TwoPartResult, score: _ScoreBase):
    score._structural[OuterVoice.BASS].append(pitches["bass"])
    score._structural[OuterVoice.MELODY].append(pitches["melody"])


def pop_structural_pitches(score: _ScoreBase):
    score._structural[OuterVoice.BASS].pop()
    score._structural[OuterVoice.MELODY].pop()
