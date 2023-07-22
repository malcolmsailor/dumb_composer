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
    IntervalChooser2,
    IntervalChooser2Settings,
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
from dumb_composer.pitch_utils.types import ChromaticInterval, Pitch, TimeStamp, Weight
from dumb_composer.shared_classes import Annotation, Score, _ScoreBase
from dumb_composer.suspensions import (
    Suspension,
    find_bass_suspension,
    find_suspension_release_times,
    find_suspensions,
)
from dumb_composer.time import Meter
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf
from dumb_composer.utils.math_ import softmax, weighted_sample_wo_replacement
from dumb_composer.utils.recursion import DeadEnd

LOGGER = logging.getLogger(__name__)


class OuterVoice(IntEnum):
    BASS = 0
    MELODY = 1


@dataclass
class TwoPartContrapuntistSettings(IntervalChooser2Settings):
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
    annotate_suspensions: bool = False  # TODO: (Malcolm 2023-07-21) restore True
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
        # TODO the size of lambda parameter for IntervalChooser should depend on
        #   how long the chord is. If a chord lasts for a whole note it can move
        #   by virtually any amount. If a chord lasts for an eighth note it
        #   should move by a relatively small amount.
        # TODO: (Malcolm 2023-07-17) and the above in turn should be influenced by the
        #   expected density of ornamentation. If we're embellishing in 16ths then
        #   each note should be free to move relatively widely.
        self._interval_chooser = IntervalChooser2(settings)
        self._suspension_resolutions: t.DefaultDict[
            OuterVoice, t.Dict[int, int]
        ] = defaultdict(dict)
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
            self._score = Score(
                chord_data, range_constraints=self.settings.range_constraints
            )
        else:
            assert score is not None
            assert score.range_constraints == self.settings.range_constraints
            self._score = score

    def validate_state(self) -> bool:
        return len(self._score.structural_melody) == len(self._score.structural_bass)

    @property
    def i(self):
        return len(self._score.structural_bass)

    # -----------------------------------------------------------------------------------
    # Properties and helper functions
    # -----------------------------------------------------------------------------------

    @property
    def empty(self) -> bool:
        assert self.validate_state()
        return len(self._score.structural_bass) == 0

    @property
    def complete(self) -> bool:
        assert self.validate_state()
        return len(self._score.structural_bass) == len(self._score.chords)

    @property
    def ts(self) -> Meter:
        return self._score.ts

    @property
    def current_chord(self) -> Chord:
        return self._score.chords[self.i]

    @property
    def prev_chord(self) -> Chord | None:
        if self.i <= 0:
            return None
        return self._score.chords[self.i - 1]

    @property
    def next_chord(self) -> Chord | None:
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.chords[self.i + 1]

    @property
    def prev_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i < 1:
            raise ValueError()
        return self._score.pc_bass[self.i - 1]

    @property
    def current_foot_pc(self) -> PitchClass:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        return self._score.pc_bass[self.i]

    @property
    def next_foot_pc(self) -> PitchClass | None:
        """The "notional" bass pitch-class.

        It is "notional" because the pitch-class of the *actual* bass may be a
        suspension, etc. The "foot" differs from the "root" in that it isn't necessarily
        the root. E.g., in a V6 chord in C major, with a bass suspension C--B, on the C,
        the foot is the third B.
        """
        if self.i + 1 >= len(self._score.chords):
            return None
        return self._score.pc_bass[self.i + 1]

    @property
    def prev_bass_pitch(self) -> Pitch:
        if self.i < 1:
            raise ValueError()
        return self._score.structural_bass[self.i - 1]

    @property
    def prev_melody_pitch(self) -> Pitch:
        if self.i < 1:
            raise ValueError()
        return self._score.structural_melody[self.i - 1]

    @property
    def prev_structural_pitch(self) -> dict[OuterVoice, Pitch]:
        if self.i < 1:
            raise ValueError()
        return {
            OuterVoice.BASS: self._score.structural_bass[self.i - 1],
            OuterVoice.MELODY: self._score.structural_melody[self.i - 1],
        }

    def add_suspension_resolution(self, voice: OuterVoice, pitch: Pitch):
        assert not self.i + 1 in self._suspension_resolutions[voice]
        self._suspension_resolutions[voice][self.i + 1] = pitch

    def remove_suspension_resolution(self, voice: OuterVoice):
        assert self.i + 1 in self._suspension_resolutions[voice]
        del self._suspension_resolutions[voice][self.i + 1]

    def suspension_in_other_voice(self, voice: OuterVoice) -> bool:
        if voice is OuterVoice.MELODY:
            return self.bass_suspension is not None
        if voice is OuterVoice.BASS:
            return self.melody_suspension is not None
        raise ValueError()

    @property
    def has_melody_suspension(self) -> bool:
        return self.i in self._score.melody_suspensions

    @property
    def has_bass_suspension(self) -> bool:
        return self.i in self._score.bass_suspensions

    def has_suspension_resolution(self, voice: OuterVoice) -> bool:
        return self.i in self._suspension_resolutions[voice]

    @property
    def melody_suspension(self) -> Suspension | None:
        if self.i not in self._score.melody_suspensions:
            return None
        return self._score.melody_suspensions[self.i]

    @property
    def bass_suspension(self) -> Suspension | None:
        if self.i not in self._score.bass_suspensions:
            return None
        return self._score.bass_suspensions[self.i]

    @property
    def prev_melody_suspension(self) -> Suspension | None:
        if self.i - 1 not in self._score.melody_suspensions:
            return None
        return self._score.melody_suspensions[self.i - 1]

    @property
    def prev_bass_suspension(self) -> Suspension | None:
        if self.i - 1 not in self._score.bass_suspensions:
            return None
        return self._score.bass_suspensions[self.i - 1]

    def add_suspension(self, voice: OuterVoice, suspension: Suspension):
        if voice is OuterVoice.BASS:
            assert self.i not in self._score.bass_suspensions
            self._score.bass_suspensions[self.i] = suspension
        elif voice is OuterVoice.MELODY:
            assert self.i not in self._score.melody_suspensions
            self._score.melody_suspensions[self.i] = suspension
        else:
            raise ValueError()

    def remove_suspension(self, voice: OuterVoice):
        if voice is OuterVoice.BASS:
            self._score.bass_suspensions.pop(self.i)  # raises KeyError if not present
        elif voice is OuterVoice.MELODY:
            self._score.melody_suspensions.pop(self.i)  # raises KeyError if not present
        else:
            raise ValueError()

    def annotate_suspension(self, voice: OuterVoice):
        annotations_label = (
            "bass_suspensions" if voice is OuterVoice.BASS else "melody_suspensions"
        )
        self._score.annotations[annotations_label].append(
            Annotation(self.current_chord.onset, "S")
        )

    def remove_suspension_annotation(self, voice: OuterVoice):
        annotations_label = (
            "bass_suspensions" if voice is OuterVoice.BASS else "melody_suspensions"
        )
        popped_annotation = self._score.annotations[annotations_label].pop()
        assert popped_annotation.onset == self.current_chord.onset

    def split_current_chord_at(self, time: TimeStamp):
        LOGGER.debug(
            f"Splitting chord at {time=} w/ duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )
        self._score.split_ith_chord_at(self.i, time)
        LOGGER.debug(
            f"After split chord has duration "
            f"{self.current_chord.release - self.current_chord.onset}"
        )
        self._split_chords.append(self.i)

    def merge_current_chords_if_they_were_previously_split(self):
        if self._split_chords and self._split_chords[-1] == self.i:
            assert self.next_chord is not None
            assert is_same_harmony(self.current_chord, self.next_chord)
            self._score.merge_ith_chords(self.i)
            self._split_chords.pop()

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

        # MELODY ONLY
        if self.bass_suspension:
            bass_suspension_pitch = self.bass_suspension.pitch
        else:
            bass_suspension_pitch = None

        # Here we depend on the fact that
        #   1. self.next_chord.onset is the greatest possible time that can occur
        #       in release_times
        #   2. release_times are sorted from greatest to least.
        # Thus, if self.next_chord.onset is in release_times, it is the first
        #   element.
        if self.next_chord is not None and self.next_chord.onset == release_times[0]:
            next_chord_release_time = self.next_chord.onset
            release_times = release_times[1:]

            # MELODY ONLY
            next_foot_pc: PitchClass = self.next_foot_pc  # type:ignore
            next_foot_tendency = self.next_chord.get_pitch_tendency(next_foot_pc)
            resolution_chord_pcs_to_avoid: set[PitchClass] = (
                set() if next_foot_tendency is Tendency.NONE else {next_foot_pc}
            )

            next_chord_suspensions = find_suspensions(
                self.prev_melody_pitch,
                suspension_chord=self.current_chord,
                resolution_chord=self.next_chord,
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
                self.prev_melody_pitch,
                suspension_chord=self.current_chord,
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
                        release_time - self.current_chord.onset  # type:ignore
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
        if self.current_chord.foot == self.prev_foot_pc:
            # if the current chord bass is the same as the previous bass, return
            # immediately
            yield None
            return

        release_times = find_suspension_release_times(
            self.current_chord.onset,
            self.current_chord.release,
            self._score.ts,
            max_weight_diff=self.settings.max_suspension_weight_diff,
            max_suspension_dur=self.settings.max_suspension_dur,
            include_stop=self.next_chord is not None,
        )
        if not release_times:
            yield None
            return

        if self.melody_suspension:
            other_suspension_pitches = (self.melody_suspension.pitch,)
        else:
            other_suspension_pitches = ()

        # Here we depend on the fact that
        #   1. self.next_chord.onset is the greatest possible time that can occur
        #       in release_times
        #   2. release_times are sorted from greatest to least.
        # Thus, if self.next_chord.onset is in release_times, it is the first
        #   element.
        if self.next_chord is not None and self.next_chord.onset == release_times[0]:
            next_chord_release_time = self.next_chord.onset
            release_times = release_times[1:]

            # BASS ONLY
            next_chord_suspensions = find_bass_suspension(
                src_pitch=self.prev_bass_pitch,
                suspension_chord=self.current_chord,
                resolution_chord=self.next_chord,
                resolve_up_by=self._upward_suspension_resolutions_bass,
                other_suspended_pitches=other_suspension_pitches,
            )
        else:
            next_chord_release_time = TimeStamp(0)
            next_chord_suspensions = []

        if release_times:
            # BASS ONLY
            suspensions = find_bass_suspension(
                src_pitch=self.prev_bass_pitch,
                suspension_chord=self.current_chord,
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
                        release_time - self.current_chord.onset
                        for release_time in release_times
                    ]
                    # We sample suspension releases directly proportional to the
                    #   resulting suspension length.
                    suspension_release = random.choices(
                        release_times, k=1, weights=suspension_lengths
                    )[0]
                    yield suspension, suspension_release

    def _apply_suspension(
        self,
        suspension: Suspension,
        suspension_release: TimeStamp,
        voice: OuterVoice = OuterVoice.MELODY,
    ) -> Pitch:
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

        suspended_pitch = self.prev_structural_pitch[voice]
        self.add_suspension_resolution(voice, suspended_pitch + suspension.resolves_by)
        self.add_suspension(voice, suspension)
        if self.settings.annotate_suspensions:
            self.annotate_suspension(voice)
        return suspended_pitch

    def _undo_suspension(self, voice: OuterVoice = OuterVoice.MELODY) -> None:
        if not self.suspension_in_other_voice(voice):
            self.merge_current_chords_if_they_were_previously_split()
        self.remove_suspension_resolution(voice)
        self.remove_suspension(voice)
        if self.settings.annotate_suspensions:
            self.remove_suspension_annotation(voice)

    def _get_tendency(
        self, intervals: t.List[int], bass_has_tendency: bool
    ) -> t.Iterable[int]:
        assert self.prev_chord is not None
        tendency = self.prev_chord.get_pitch_tendency(self.prev_melody_pitch)
        if tendency is Tendency.NONE or self.prev_chord == self.current_chord:
            return
        if tendency is Tendency.UP:
            steps = (1, 2)
        else:
            steps = (-1, -2)
        for step in steps:
            if step in intervals:
                candidate_pitch = self.prev_melody_pitch + step
                if bass_has_tendency and (candidate_pitch % 12 == self.current_foot_pc):
                    return
                yield candidate_pitch
                intervals.remove(step)
                return
        LOGGER.debug(
            f"resolving tendency-tone with pitch {self.prev_melody_pitch} "
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
        eligible_pcs = self.current_chord.get_pcs_that_can_be_added_to_existing_voicing(
            (self.current_foot_pc,)
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
            self.current_foot_pc, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _resolve_suspension(self, voice: OuterVoice = OuterVoice.MELODY):
        yield self._suspension_resolutions[voice][self.i]

    # def _get_lingering_resolution_weights(
    #     self,
    #     prev_pitch: Pitch,
    #     candidate_intervals: t.Iterable[ChromaticInterval],
    # ) -> None | t.List[Weight]:
    #     lingering_tendencies = self._lingering_tendencies[-1]

    #     out = []

    #     for candidate_interval in candidate_intervals:
    #         candidate_pitch = prev_pitch + candidate_interval
    #         out.append(lingering_tendencies.get(candidate_pitch, 0.0))

    #     return out

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

        # if voice_to_choose_for is OuterVoice.MELODY:
        #     custom_weights = self._get_lingering_resolution_weights(
        #         prev_pitch, intervals
        #     )
        # else:
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
        foot_tendency = self.current_chord.get_pitch_tendency(self.current_foot_pc)
        dont_double_bass_pc = foot_tendency is not Tendency.NONE

        # "no suspension" is understood as `suspension is None`
        for suspension in self._choose_suspension(
            suspension_chord_pcs_to_avoid={self.current_foot_pc}
            if dont_double_bass_pc
            else set()
        ):
            self._lingering_tendencies.append({})
            if suspension is not None:
                yield self._apply_suspension(*suspension)
                # we only continue execution if the recursive calls fail
                #   somewhere further down the stack, in which case we need to
                #   undo the suspension.
                self._undo_suspension()
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if current_bass_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        self.prev_melody_pitch,
                        [(self.prev_bass_pitch, current_bass_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.MELODY, pitch_in_other_voice=current_bass_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    self.prev_melody_pitch,
                    self.current_chord.pcs,
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
                        f"{self.__class__.__name__} yielding tendency resolution pitch {pitch}"
                    )

                    yield pitch
                    # self._lingering_tendencies[i + 1][
                    #     pitch
                    # ] = self.settings.tendency_decay_per_measure

                # If the previous pitch does not have a tendency, or proceeding
                # according to the tendency doesn't work, try the other intervals
                yield from self._choose_intervals(
                    self.prev_bass_pitch,
                    current_bass_pitch,
                    self.prev_melody_pitch,
                    dont_double_bass_pc,
                    intervals,
                    voice_to_choose_for=OuterVoice.MELODY,
                    notional_other_pc=self.current_foot_pc,
                )
            # self._lingering_tendencies.pop()

    def _get_bass_pitch(self, current_melody_pitch: Pitch | None) -> t.Iterator[Pitch]:
        # I'm not sure whether it's strictly necessary to calculate whether we can
        # double the melody pc, since the bass foot pc is determined and the melody
        # should not double it when not appropriate to do so.
        if current_melody_pitch is None:
            dont_double_melody_pc = False
        elif self.has_melody_suspension:
            # Don't double a suspension
            dont_double_melody_pc = True
        else:
            # Don't double the pc if it is a tendency tone
            dont_double_melody_pc = (
                self.current_chord.get_pitch_tendency(current_melody_pitch)
                is not Tendency.NONE
            )

        for suspension in self._choose_suspension_bass():
            if suspension is not None:
                yield self._apply_suspension(*suspension, voice=OuterVoice.BASS)

                # we only continue execution if the recursive calls fail
                #   somewhere further down the stack, in which case we need to
                #   undo the suspension.
                self._undo_suspension(voice=OuterVoice.BASS)
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if current_melody_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        self.prev_bass_pitch,
                        [(self.prev_melody_pitch, current_melody_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.BASS, pitch_in_other_voice=current_melody_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    self.prev_bass_pitch,
                    (self.current_chord.foot,),
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
                    self.prev_melody_pitch,
                    current_melody_pitch,
                    self.prev_bass_pitch,
                    dont_double_melody_pc,
                    intervals,
                    voice_to_choose_for=OuterVoice.MELODY,
                )

    def _melody_step(self, bass_pitch: Pitch | None = None) -> t.Iterator[Pitch]:
        # -------------------------------------------------------------------------------
        # Condition 1: start the melody
        # -------------------------------------------------------------------------------

        if self.empty:
            # generate the first note
            for pitch in self._get_first_melody_pitch(bass_pitch):
                LOGGER.debug(f"{self.__class__.__name__} yielding first pitch {pitch}")
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        elif self.has_suspension_resolution(OuterVoice.MELODY):
            for pitch in self._resolve_suspension():
                LOGGER.debug(
                    f"{self.__class__.__name__} yielding suspension resolution pitch {pitch}"
                )
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            yield from self._get_melody_pitch(bass_pitch)

    def _bass_step(self, melody_pitch: Pitch | None = None) -> t.Iterator[Pitch]:
        # -------------------------------------------------------------------------------
        # Condition 1: start the bass
        # -------------------------------------------------------------------------------
        if self.empty:
            for pitch in self._get_first_bass_pitch(melody_pitch):
                LOGGER.debug(f"{self.__class__.__name__} yielding first pitch {pitch}")
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        elif self.has_suspension_resolution(OuterVoice.BASS):
            for pitch in self._resolve_suspension(OuterVoice.BASS):
                LOGGER.debug(
                    f"{self.__class__.__name__} yielding bass suspension resolution pitch {pitch}"
                )
                yield pitch
        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            for pitch in self._get_bass_pitch(melody_pitch):
                yield pitch

    def _step(self) -> t.Iterator[t.Dict[str, Pitch]]:
        self.validate_state()
        if self.settings.do_first is OuterVoice.BASS:
            for bass_pitch in self._bass_step():
                for melody_pitch in self._melody_step(bass_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}
        else:
            for melody_pitch in self._melody_step():
                for bass_pitch in self._bass_step(melody_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}

        LOGGER.debug("reached dead end")
        raise DeadEnd()

    def __call__(self):
        assert self.empty
        while not self.complete:
            pitches = next(self._step())
            self._score.structural_melody.append(pitches["melody"])
            self._score.structural_bass.append(pitches["bass"])
        return self._score

    def get_mididf_from_score(self, score: _ScoreBase):
        out_dict = {
            "onset": [chord.onset for chord in score.chords],
            "release": [chord.release for chord in score.chords],
            "bass": score.structural_bass,
            "melody": score.structural_melody,
        }
        out_df = pd.DataFrame(out_dict)
        return homodf_to_mididf(out_df)
