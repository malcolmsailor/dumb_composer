import logging
import random
import typing as t
from dataclasses import dataclass
from itertools import chain, product, repeat
from statistics import mean

from dumb_composer.pitch_utils.chords import Chord, Tendency
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
    ALTO,
    BASS,
    MELODY,
    TENOR,
    TENOR_AND_ALTO,
    ChromaticInterval,
    InnerVoice,
    OuterVoice,
    Pitch,
    TimeStamp,
    Voice,
    VoicePair,
)
from dumb_composer.pitch_utils.voice_lead_chords import voice_lead_chords
from dumb_composer.shared_classes import ScoreInterface, _ScoreBase
from dumb_composer.suspensions import (
    Suspension,
    SuspensionCombo,
    find_bass_suspension,
    find_suspension_release_times,
    find_suspensions,
    validate_intervals_among_suspensions,
    validate_suspension_resolution,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.math_ import softmax, weighted_sample_wo_replacement
from dumb_composer.utils.recursion import StructuralDeadEnd, recursive_attempt

LOGGER = logging.getLogger(__name__)

# TODO: (Malcolm 2023-08-07) find and execute voice exchanges, esp. between outer voices
#   (note that presently stepwise voice exchanges are if anything discouraged because of
#   the octaves)


@dataclass
class IncrementalContrapuntistSettings(IntervalChooserSettings):
    forbidden_parallels: t.Sequence[int] = (7, 0)
    forbidden_antiparallels: t.Sequence[int] = (0,)
    unpreferred_direct_intervals: t.Sequence[int] = (7, 0)
    max_interval: int = 12
    max_suspension_weight_diff: int = 1
    max_suspension_dur: t.Union[TimeStamp, str] = "bar"
    annotate_suspensions: bool = False  # TODO: (Malcolm 2023-07-25) restore
    allow_steps_outside_of_range: bool = True
    # when choosing whether to insert a suspension, we put the "score" of each
    #   suspension (so far, by default 1.0) into a softmax together with the
    #   following "no_suspension_score".
    # To ensure that suspensions will be used wherever possible,
    #   `no_suspension_score` can be set to a large negative number (which
    #   will become zero after the softmax) or even float("-inf").
    no_suspension_score: float = 2.0
    allow_avoid_intervals: bool = False
    weight_harmonic_intervals: bool = True

    inner_voice_suspensions_dont_cross_melody: bool = True

    range_constraints: RangeConstraints = RangeConstraints()
    spacing_constraints: SpacingConstraints = SpacingConstraints()


IncrementalResult = dict[Voice, Pitch]


def _validate(voices: t.Sequence[Voice]):
    def f(score):
        out_lens = []
        for voice in voices:
            if voice is TENOR_AND_ALTO:
                out_lens.append(len(score._structural[TENOR]))
                out_lens.append(len(score._structural[ALTO]))
            else:
                out_lens.append(len(score._structural[voice]))
        return len(set(out_lens)) == 1

    return f


class IncrementalContrapuntist:
    def __init__(
        self,
        *,
        score: _ScoreBase,
        voices: t.Sequence[Voice],
        prior_voices: t.Sequence[Voice] = (),
        settings: IncrementalContrapuntistSettings = IncrementalContrapuntistSettings(),
    ):
        self.settings = settings
        # TODO: (Malcolm 2023-08-04) can we update IntervalChooser -> DeprecatedIntervalChooser
        self._interval_chooser = IntervalChooser(settings)

        self._spinner = Spinner()
        self._prior_voices = tuple(prior_voices)
        self._voices = tuple(voices)
        self._validate_voices(self._prior_voices + self._voices)

        # TODO: (Malcolm 2023-08-08) is this a robust way of setting get_i?
        get_i = lambda score: len(score._structural[voices[0]])

        self._score = ScoreInterface(
            score, get_i=get_i, validate=_validate(self._prior_voices + self._voices)
        )

    # ==================================================================================
    # Utilities
    # ==================================================================================

    def _upward_suspension_resolutions(
        self, voice: Voice
    ) -> tuple[ChromaticInterval, ...]:
        # TODO: (Malcolm 2023-08-04)
        return ()

    def _get_outervoice_boundary_pitches(
        self, voice: OuterVoice, pitch_in_other_voice: Pitch | None = None
    ) -> tuple[Pitch, Pitch]:
        if voice is BASS:
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

        elif voice is MELODY:
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

    # ==================================================================================
    # Melody
    # ==================================================================================

    def _get_melody_tendency(
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

    def _choose_outervoice_intervals(
        self,
        src_other_pitch: Pitch | None,
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
                assert src_other_pitch is not None
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
                if voice_to_choose_for is BASS:
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

    def _choose_melody_suspension(
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
        bass_suspension = self._score.current_suspension(BASS)
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
                resolve_up_by=self._upward_suspension_resolutions(MELODY),
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
                resolve_up_by=self._upward_suspension_resolutions(MELODY),
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

    def _get_first_melody_pitch(self, bass_pitch: Pitch | None) -> t.Iterator[Pitch]:
        eligible_pcs = (
            self._score.current_chord.get_pcs_that_can_be_added_to_existing_voicing(
                (self._score.current_foot_pc,)
            )
        )

        min_pitch, max_pitch = self._get_outervoice_boundary_pitches(
            MELODY, pitch_in_other_voice=bass_pitch
        )
        yield from get_all_in_range(
            eligible_pcs, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _get_melody_pitch(self, current_bass_pitch: Pitch | None) -> t.Iterator[Pitch]:
        foot_tendency = self._score.current_chord.get_pitch_tendency(
            self._score.current_foot_pc
        )
        dont_double_bass_pc = foot_tendency is not Tendency.NONE

        # "no suspension" is understood as `suspension is None`
        for suspension in self._choose_melody_suspension(
            suspension_chord_pcs_to_avoid={self._score.current_foot_pc}
            if dont_double_bass_pc
            else set()
        ):
            if suspension is not None:
                yield self._score.apply_suspension(
                    *suspension,
                    voice=MELODY,
                    annotate=self.settings.annotate_suspensions,
                )
                # we only continue execution if the recursive calls fail
                #   somewhere further down the stack, in which case we need to
                #   undo the suspension.
                self._score.undo_suspension(
                    voice=MELODY,
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
                min_pitch, max_pitch = self._get_outervoice_boundary_pitches(
                    MELODY, pitch_in_other_voice=current_bass_pitch
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
                for pitch in self._get_melody_tendency(intervals, dont_double_bass_pc):
                    # self._get_tendency removes intervals
                    LOGGER.debug(
                        f"{self.__class__.__name__} using tendency resolution pitch {pitch}"
                    )
                    yield pitch

                try:
                    prev_bass_pitch = self._score.prev_pitch(BASS)
                except IndexError:
                    # This will occur when we are doing one part only, the melody.
                    # Maybe there's a more elegant way than catching the IndexError
                    # every time.
                    prev_bass_pitch = None

                # If the previous pitch does not have a tendency, or proceeding
                # according to the tendency doesn't work, try the other intervals
                yield from self._choose_outervoice_intervals(
                    prev_bass_pitch,
                    current_bass_pitch,
                    self._score.prev_pitch(MELODY),
                    dont_double_bass_pc,
                    intervals,
                    voice_to_choose_for=MELODY,
                    notional_other_pc=self._score.current_foot_pc,
                )

    def _melody_step(self, bass_pitch: Pitch | None) -> t.Iterator[Pitch]:
        if bass_pitch is None:
            # If there is no bass pitch in progress, we check if bass pitch
            #   has already been saved to the score
            bass_pitch = self._score.current_pitch(BASS)
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
        elif (pitch := self._score.current_resolution(MELODY)) is not None:
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

    # ==================================================================================
    # Bass
    # ==================================================================================
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
                resolve_up_by=self._upward_suspension_resolutions(BASS),
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
                resolve_up_by=self._upward_suspension_resolutions(BASS),
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

    def _get_first_bass_pitch(self, melody_pitch: Pitch | None) -> t.Iterator[Pitch]:
        min_pitch, max_pitch = self._get_outervoice_boundary_pitches(
            BASS, pitch_in_other_voice=melody_pitch
        )

        yield from get_all_in_range(
            self._score.current_foot_pc, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _get_bass_pitch(self, current_melody_pitch: Pitch | None) -> t.Iterator[Pitch]:
        # I'm not sure whether it's strictly necessary to calculate whether we can
        # double the melody pc, since the bass foot pc is determined and the melody
        # should not double it when not appropriate to do so.
        if current_melody_pitch is None:
            dont_double_melody_pc = False
        elif self._score.current_suspension(MELODY):
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
                    voice=BASS,
                    annotate=self.settings.annotate_suspensions,
                )
                # Note: since we yield above (rather than making a recursive call), we
                #   need to be careful to undo the suspension if the recursive calls
                #   fail.
                self._score.undo_suspension(
                    voice=BASS,
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
                min_pitch, max_pitch = self._get_outervoice_boundary_pitches(
                    BASS, pitch_in_other_voice=current_melody_pitch
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

                try:
                    prev_melody_pitch = self._score.prev_pitch(MELODY)
                except IndexError:
                    # This will occur when we are doing one part only, the bass.
                    # Maybe there's a more elegant way than catching the IndexError
                    # every time.
                    prev_melody_pitch = None

                yield from self._choose_outervoice_intervals(
                    prev_melody_pitch,
                    current_melody_pitch,
                    self._score.prev_pitch(BASS),
                    dont_double_melody_pc,
                    intervals,
                    voice_to_choose_for=BASS,
                )

    def _bass_step(self, melody_pitch: Pitch | None = None) -> t.Iterator[Pitch]:
        if melody_pitch is None:
            # If there is no bass pitch in progress, we check if bass pitch
            #   has already been saved to the score
            melody_pitch = self._score.current_pitch(MELODY)
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

        elif (pitch := self._score.current_resolution(BASS)) is not None:
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

    # ==================================================================================
    # Inner voices
    # ==================================================================================

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
            for tenor_suspension, alto_suspension in product(
                tenor_suspensions, alto_suspensions
            )
        ]
        return tenor_suspensions + alto_suspensions + both_voice_suspensions

    def _get_valid_suspension_combinations(
        self,
        suspensions_per_inner_voice: dict[InnerVoice, list[Suspension]],
        voice_or_voices: InnerVoice | VoicePair,
    ) -> list[SuspensionCombo]:
        if not suspensions_per_inner_voice:
            return []
        if voice_or_voices is not TENOR_AND_ALTO:
            return [
                {voice_or_voices: suspension}
                for suspension in suspensions_per_inner_voice[voice_or_voices]
            ]
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

    def _choose_inner_voice_suspensions(
        self,
        voice_or_voices: InnerVoice | VoicePair,
        melody_pitch: Pitch,
        bass_pitch: Pitch,
        free_inner_voices: t.Sequence[InnerVoice],
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
                    resolve_up_by=self._upward_suspension_resolutions(voice_or_voices),
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
                    resolve_up_by=self._upward_suspension_resolutions(voice_or_voices),
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

        suspension_combos = self._get_valid_suspension_combinations(
            suspensions, voice_or_voices
        )
        next_chord_suspension_combos = self._get_valid_suspension_combinations(
            next_chord_suspensions, voice_or_voices
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
            list(zip(suspension_combos, repeat(release_times)))
            + list(zip(next_chord_suspension_combos, repeat([next_chord_release_time])))
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

    def _resolve_inner_voice_suspension_step(
        self, voice: InnerVoice, melody_pitch: Pitch, bass_pitch: Pitch
    ) -> Pitch | None:
        if not self._validate_suspension_resolution(
            voice, melody_pitch=melody_pitch, bass_pitch=bass_pitch
        ):
            # If the melody doubles a resolution pc, the validation will fail.
            # It would be more efficient but rather more work to not generate the
            # melody in the first place.
            return

        # TODO: (Malcolm 2023-08-04) if we call this function current_resolution()
        #   should not be None. Remove this comment eventually.
        out = self._score.current_resolution(voice)
        assert out is not None
        return out

    def _voice_lead_chords(
        self,
        voice_or_voices: InnerVoice | VoicePair,
        melody_pitch: Pitch,
        bass_pitch: Pitch,
        tenor_resolution_pitch: Pitch | None = None,
        alto_resolution_pitch: Pitch | None = None,
    ) -> t.Iterator[tuple[Pitch, ...]]:
        # -------------------------------------------------------------------------------
        # Calculate free inner voices
        # -------------------------------------------------------------------------------
        if voice_or_voices is TENOR and alto_resolution_pitch:
            raise ValueError()
        if voice_or_voices is ALTO and tenor_resolution_pitch:
            raise ValueError()

        if voice_or_voices is TENOR_AND_ALTO:
            free_inner_voices = [TENOR, ALTO]
        else:
            free_inner_voices = [voice_or_voices]

        if tenor_resolution_pitch is not None:
            free_inner_voices.remove(TENOR)
        if alto_resolution_pitch is not None:
            free_inner_voices.remove(ALTO)

        if not free_inner_voices:
            yield tuple(
                p
                for p in (tenor_resolution_pitch, alto_resolution_pitch)
                if p is not None
            )
            return

        # ------------------------------------------------------------------------------
        # calculate suspensions
        # ------------------------------------------------------------------------------
        chord1_suspensions = {}
        chord2_suspensions = {}

        for voice in chain(OuterVoice, InnerVoice):
            prev_suspension = self._score.prev_suspension(voice)
            if prev_suspension is not None:
                chord1_suspensions[prev_suspension.pitch] = prev_suspension

            suspension = self._score.current_suspension(voice)
            if suspension is not None:
                chord2_suspensions[suspension.pitch] = suspension

        # ------------------------------------------------------------------------------
        # helper function
        # ------------------------------------------------------------------------------
        def _result_from_voicing(
            voicing: tuple[Pitch, ...], suspension_combo: SuspensionCombo | None = None
        ) -> tuple[Pitch, ...]:
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
                return (tenor_resolution_pitch, other_pitch)

            elif alto_resolution_pitch is not None:
                other_pitch_i = int(pitch_pair[0] == alto_resolution_pitch)
                other_pitch = pitch_pair[other_pitch_i]
                assert suspension_voice is None or suspension_voice is InnerVoice.TENOR
                assert suspension is None or suspension.pitch == other_pitch
                return (other_pitch, alto_resolution_pitch)

            elif suspension is not None:
                other_pitch_i = int(pitch_pair[0] == suspension.pitch)
                other_pitch = pitch_pair[other_pitch_i]
                if suspension_voice is InnerVoice.TENOR:
                    return (suspension.pitch, other_pitch)
                else:
                    return (other_pitch, suspension.pitch)

            # if not pitch_pair:
            #     breakpoint()

            return pitch_pair

        for suspensions in self._choose_inner_voice_suspensions(
            voice_or_voices, melody_pitch, bass_pitch, free_inner_voices
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
                if len(suspension_combo) == len(free_inner_voices):
                    assert not alto_resolution_pitch or not tenor_resolution_pitch
                    yield tuple(
                        suspension_combo[voice].pitch
                        for voice in (TENOR, ALTO)
                        if voice in suspension_combo
                    )
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

    def _inner_voice_step(
        self,
        voice_or_voices: InnerVoice | VoicePair,
        melody_pitch: Pitch | None,
        bass_pitch: Pitch | None,
    ) -> t.Iterator[tuple[Pitch, ...]]:
        if melody_pitch is None:
            # If there is no bass pitch in progress, we check if bass pitch
            #   has already been saved to the score
            melody_pitch = self._score.current_pitch(MELODY)
        if bass_pitch is None:
            # If there is no bass pitch in progress, we check if bass pitch
            #   has already been saved to the score
            bass_pitch = self._score.current_pitch(BASS)
        assert melody_pitch is not None and bass_pitch is not None

        # -------------------------------------------------------------------------------
        # Condition 1: start the inner voices
        # -------------------------------------------------------------------------------
        if self._score.empty:
            n_notes = 4 if voice_or_voices is TENOR_AND_ALTO else 3
            for pitch_voicing in self._score.current_chord.pitch_voicings(
                min_notes=n_notes,
                max_notes=n_notes,
                melody_pitch=melody_pitch,
                bass_pitch=bass_pitch,
                range_constraints=self.settings.range_constraints,
                spacing_constraints=self.settings.spacing_constraints,
                shuffled=True,
            ):
                inner_voices = pitch_voicing[1:-1]
                LOGGER.debug("yielding initial {inner_voices=}")
                yield inner_voices
                return

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        tenor_resolution_pitch = None
        alto_resolution_pitch = None
        if voice_or_voices in (TENOR, TENOR_AND_ALTO):
            tenor_resolves = self._score.current_resolution(TENOR) is not None
            if tenor_resolves:
                tenor_resolution_pitch = self._resolve_inner_voice_suspension_step(
                    TENOR, melody_pitch, bass_pitch
                )
                if tenor_resolution_pitch is None:
                    return
        if voice_or_voices in (ALTO, TENOR_AND_ALTO):
            alto_resolves = self._score.current_resolution(ALTO) is not None
            if alto_resolves:
                alto_resolution_pitch = self._resolve_inner_voice_suspension_step(
                    ALTO, melody_pitch, bass_pitch
                )
                if alto_resolution_pitch is None:
                    return

        # -------------------------------------------------------------------------------
        # Condition 3: voice-lead the inner-voices freely
        # -------------------------------------------------------------------------------
        for item in self._voice_lead_chords(
            voice_or_voices,
            melody_pitch,
            bass_pitch,
            tenor_resolution_pitch,
            alto_resolution_pitch,
        ):
            if not item:
                breakpoint()
            yield item
        LOGGER.debug("no more inner voices")

    # ==================================================================================
    # Main logic
    # ==================================================================================
    def _handle_step(
        self,
        voice_or_voices: Voice,
        result_so_far: IncrementalResult,
    ) -> t.Iterator[Pitch | tuple[Pitch]]:
        if voice_or_voices is BASS:
            in_progress_melody_pitch = result_so_far.get(MELODY, None)
            yield from self._bass_step(melody_pitch=in_progress_melody_pitch)
        elif voice_or_voices is MELODY:
            in_progress_bass_pitch = result_so_far.get(BASS, None)
            yield from self._melody_step(bass_pitch=in_progress_bass_pitch)
        elif voice_or_voices in (TENOR, ALTO, TENOR_AND_ALTO):
            in_progress_melody_pitch = result_so_far.get(MELODY, None)
            in_progress_bass_pitch = result_so_far.get(BASS, None)
            iter = self._inner_voice_step(
                voice_or_voices, in_progress_melody_pitch, in_progress_bass_pitch
            )
            if voice_or_voices is TENOR_AND_ALTO:
                yield from iter
            else:
                for tup in iter:
                    yield tup[0]
        else:
            raise ValueError

    def _validate_voices(self, voices: t.Sequence[Voice]):
        assert len(set(voices)) == len(voices)
        saw_bass = False
        saw_melody = False
        saw_inner_voices = False
        for voice in voices:
            if voice is BASS:
                saw_bass = True
            elif voice is MELODY:
                saw_melody = True
            elif voice in (ALTO, TENOR, TENOR_AND_ALTO):
                if saw_inner_voices:
                    raise ValueError(
                        "To do both alto and tenor, use the TENOR_AND_ALTO enum and "
                        "not ALTO and/or TENOR"
                    )
                if not saw_bass or not saw_melody:
                    raise ValueError(
                        "Inner voices can only be created after both BASS and MELODY"
                    )
                saw_inner_voices = True
        if not (saw_bass or saw_melody):
            raise ValueError(
                "There should be at least one voice, either the bass or melody"
            )

    def _step(
        self,
        voices: t.Sequence[Voice],
        result_so_far: IncrementalResult | None = None,
    ) -> t.Iterator[IncrementalResult]:
        if result_so_far is None:
            result_so_far = {}
        if not voices:
            yield result_so_far
            return

        voice_or_voices, *remaining_voices = voices
        for pitch_or_pitches in self._handle_step(voice_or_voices, result_so_far):
            if voice_or_voices is TENOR_AND_ALTO:
                new_result = dict(zip((TENOR, ALTO), pitch_or_pitches))  # type:ignore
            else:
                new_result = {voice_or_voices: pitch_or_pitches}
            yield from self._step(
                remaining_voices, result_so_far | new_result  # type:ignore
            )

    def _recurse(self) -> None:
        self._spinner()
        self._n_recurse_calls += 1

        if self._score.complete:
            LOGGER.debug(f"{self.__class__.__name__}._recurse: final step")
            return

        for incremental_result in self._step(self._voices):
            with recursive_attempt(
                do_func=append_structural,
                do_args=(incremental_result, self._score.score),
                undo_func=pop_structural,
                undo_args=(incremental_result, self._score.score),
            ):
                LOGGER.debug(f"append attempt {incremental_result}")
                return self._recurse()
        LOGGER.debug(f"reached dead end at {self._score.i=}")
        raise StructuralDeadEnd(
            "reached end of TwoPartContrapuntist step",  # **self._deadend_args
        )

    def __call__(self):
        assert self._score.empty
        self._n_recurse_calls = 0
        self._recurse()
        return self._score


def append_structural(incremental_result: IncrementalResult, score: _ScoreBase):
    for voice, pitch in incremental_result.items():
        assert isinstance(pitch, Pitch)
        score._structural[voice].append(pitch)


def pop_structural(voices: t.Iterable[Voice], score: _ScoreBase):
    for voice in voices:
        score._structural[voice].pop()
