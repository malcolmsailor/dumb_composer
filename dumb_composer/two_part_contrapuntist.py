import logging
import random
import sys
import typing as t
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from numbers import Number

import pandas as pd

from dumb_composer.constants import DEFAULT_BASS_RANGE, DEFAULT_MEL_RANGE
from dumb_composer.from_ml_out import get_chord_df
from dumb_composer.pitch_utils.chords import Chord, Tendency, get_chords_from_rntxt
from dumb_composer.pitch_utils.interval_chooser import (
    HarmonicallyInformedIntervalChooser,
    HarmonicallyInformedIntervalChooserSettings,
    IntervalChooser,
    IntervalChooserSettings,
)
from dumb_composer.pitch_utils.intervals import (
    get_forbidden_intervals,
    interval_finder,
    is_direct_interval,
)
from dumb_composer.pitch_utils.pcs import PitchClass
from dumb_composer.pitch_utils.put_in_range import get_all_in_range, put_in_range
from dumb_composer.pitch_utils.spacings import RangeConstraints, SpacingConstraints
from dumb_composer.pitch_utils.types import ChromaticInterval, Pitch, TimeStamp, Weight
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf

from .shared_classes import Annotation, Score, _ScoreBase
from .suspensions import Suspension, find_suspension_release_times, find_suspensions
from .utils.math_ import softmax, weighted_sample_wo_replacement
from .utils.recursion import DeadEnd

# TODO bass suspensions

# TODO don't permit suspension resolutions to tendency tones (I think I may have
# done this, double check)


class OuterVoice(IntEnum):
    BASS = 0
    MELODY = 1


@dataclass
class TwoPartContrapuntistSettings(HarmonicallyInformedIntervalChooserSettings):
    forbidden_parallels: t.Sequence[int] = (7, 0)
    forbidden_antiparallels: t.Sequence[int] = (0,)
    unpreferred_direct_intervals: t.Sequence[int] = (7, 0)
    max_interval: int = 12
    max_suspension_weight_diff: int = 1
    max_suspension_dur: t.Union[Number, str] = "bar"
    # allow_upward_suspensions can be a bool or a tuple of allowed intervals.
    # If a bool and True, the only allowed suspension is by semitone.
    allow_upward_suspensions: t.Union[bool, t.Tuple[int]] = False
    annotate_suspensions: bool = True
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

    tendency_decay_per_measure: float = 0.75

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type:ignore


# TODO: (Malcolm 2023-07-17) allow suspensions that resolve in the *following* chord
#   (perhaps only if they are sevenths and ninths?)


class TwoPartContrapuntist:
    def __init__(
        self,
        settings=None,
    ):
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
        self._ic = HarmonicallyInformedIntervalChooser(settings)
        self._suspension_resolutions: t.Dict[int, int] = {}
        self._lingering_tendencies: t.List[t.Dict[Pitch, Weight]] = []

        if not self.settings.allow_upward_suspensions:
            self._upward_suspension_resolutions = ()
        elif isinstance(self.settings.allow_upward_suspensions, bool):
            self._upward_suspension_resolutions = (1,)
        else:
            self._upward_suspension_resolutions = self.settings.allow_upward_suspensions

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

    def _choose_suspension(
        self,
        score: _ScoreBase,
        next_chord: Chord,
        cur_melody_pitch: int,
        pcs_not_to_double: t.Container[PitchClass],
    ):
        suspension_release_times = find_suspension_release_times(
            next_chord.onset,
            next_chord.release,
            score.ts,
            max_weight_diff=self.settings.max_suspension_weight_diff,
            max_suspension_dur=self.settings.max_suspension_dur,
        )
        if not suspension_release_times:
            yield None
            return

        eligible_pcs = [pc for pc in next_chord.pcs if pc not in pcs_not_to_double]

        suspensions = find_suspensions(
            # TODO providing next_scale_pcs prohibits "chromatic" suspensions
            #   (see the definition of find_suspensions()) but in the long
            #   run I'm not sure whether this is indeed something we want to
            #   do.
            cur_melody_pitch,
            eligible_pcs,
            next_scale_pcs=next_chord.scale_pcs,
            resolve_up_by=self._upward_suspension_resolutions,
        )
        if not suspensions:
            yield None
            return
        weights = softmax(
            [s.score for s in suspensions] + [self.settings.no_suspension_score]
        )
        for suspension in weighted_sample_wo_replacement(suspensions + [None], weights):
            if suspension is None:
                yield None
            else:
                # TODO suspension_release_times weights
                # TODO we only try a single release time for each suspension; should
                # we be more comprehensive?
                suspension_release = random.choices(suspension_release_times, k=1)[0]
                yield suspension, suspension_release

    def _apply_suspension(
        self,
        score: _ScoreBase,
        i: int,
        suspension: Suspension,
        suspension_release: TimeStamp,
    ) -> int:
        assert i + 1 not in self._suspension_resolutions
        score.split_ith_chord_at(i, suspension_release)
        suspended_pitch = score.structural_melody[i - 1]
        self._suspension_resolutions[i + 1] = suspended_pitch + suspension.resolves_by
        assert i not in score.melody_suspensions
        score.melody_suspensions[i] = suspension
        if self.settings.annotate_suspensions:
            score.annotations["suspensions"].append(
                Annotation(score.chords[i].onset, "S")
            )
        return suspended_pitch

    def _undo_suspension(self, score: _ScoreBase, i: int) -> None:
        assert i + 1 in self._suspension_resolutions
        score.merge_ith_chords(i)
        del self._suspension_resolutions[i + 1]
        score.melody_suspensions.pop(i)  # raises KeyError if not present
        if self.settings.annotate_suspensions:
            popped_annotation = score.annotations["suspensions"].pop()
            assert popped_annotation.onset == score.chords[i].onset

    def _get_tendency(
        self,
        score: _ScoreBase,
        i: int,
        cur_melody_pitch: int,
        next_chord: Chord,
        intervals: t.List[int],
        next_bass_has_tendency: bool,
    ) -> t.Iterable[int]:
        cur_chord = score.chords[i - 1]
        tendency = cur_chord.get_pitch_tendency(cur_melody_pitch)
        if tendency is Tendency.NONE or cur_chord == next_chord:
            return
        if tendency is Tendency.UP:
            steps = (1, 2)
        else:
            steps = (-1, -2)
        for step in steps:
            if step in intervals:
                candidate_pitch = cur_melody_pitch + step
                if next_bass_has_tendency and (
                    candidate_pitch % 12 == score.structural_bass[i] % 12
                ):
                    return
                yield candidate_pitch
                intervals.remove(step)
                return
        logging.debug(
            f"resolving tendency-tone with pitch {cur_melody_pitch} "
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

    def _get_first_melody_pitch(
        self, score: _ScoreBase, i: int, bass_pitch: Pitch | None
    ) -> t.Iterator[Pitch]:
        next_chord = score.chords[i]
        eligible_pcs = next_chord.get_pcs_that_can_be_added_to_existing_voicing()

        min_pitch, max_pitch = self._get_boundary_pitches(
            OuterVoice.MELODY, pitch_in_other_voice=bass_pitch
        )

        # TODO: (Malcolm 2023-07-14) weight choices according to interval
        yield from get_all_in_range(
            eligible_pcs, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _get_first_bass_pitch(
        self, score: _ScoreBase, i: int, melody_pitch: Pitch | None
    ) -> t.Iterator[Pitch]:
        next_bass_pc = score.pc_bass[i]

        min_pitch, max_pitch = self._get_boundary_pitches(
            OuterVoice.BASS, pitch_in_other_voice=melody_pitch
        )

        yield from get_all_in_range(
            next_bass_pc, low=min_pitch, high=max_pitch, shuffled=True
        )

    def _resolve_suspension(self, i: int):
        yield self._suspension_resolutions[i]

    # def _get_lingering_resolution_weights(
    #     self,
    #     cur_pitch: Pitch,
    #     candidate_intervals: t.Iterable[ChromaticInterval],
    # ) -> None | t.List[Weight]:
    #     lingering_tendencies = self._lingering_tendencies[-1]

    #     out = []

    #     for candidate_interval in candidate_intervals:
    #         candidate_pitch = cur_pitch + candidate_interval
    #         out.append(lingering_tendencies.get(candidate_pitch, 0.0))

    #     return out

    def _choose_intervals(
        self,
        cur_other_pitch: Pitch,
        next_other_pitch: Pitch | None,
        cur_pitch: Pitch,
        dont_double_other_pc: bool,
        intervals: t.List[ChromaticInterval],
        voice_to_choose_for: OuterVoice,
    ):
        avoid_intervals = []
        avoid_harmonic_intervals = []
        direct_intervals = []
        direct_harmonic_intervals = []

        # if voice_to_choose_for is OuterVoice.MELODY:
        #     custom_weights = self._get_lingering_resolution_weights(
        #         cur_pitch, intervals
        #     )
        # else:
        custom_weights = None

        if next_other_pitch is None:
            harmonic_intervals = None
        else:
            harmonic_intervals = [
                cur_pitch + interval - next_other_pitch for interval in intervals
            ]

        while intervals:
            interval_i = self._ic.get_interval_indices(
                intervals,
                harmonic_intervals=harmonic_intervals,
                custom_weights=custom_weights,
                n=1,
            )[0]
            interval = intervals[interval_i]
            candidate_pitch = cur_pitch + interval

            if next_other_pitch is not None:
                assert harmonic_intervals is not None
                harmonic_interval = harmonic_intervals[interval_i]
                # Check for avoid intervals

                if dont_double_other_pc and (candidate_pitch % 12) == (
                    next_other_pitch % 12
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
                        cur_pitch,
                        candidate_pitch,
                        cur_other_pitch,
                        next_other_pitch,
                    )
                else:
                    pitch_args = (
                        cur_other_pitch,
                        next_other_pitch,
                        cur_pitch,
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
            intervals.remove(interval)

        # -------------------------------------------------------------------------------
        # Try avoid intervals
        # -------------------------------------------------------------------------------
        if self.settings.allow_avoid_intervals:
            while avoid_intervals:
                interval = self._ic.get_interval_indices(
                    avoid_intervals, avoid_harmonic_intervals, n=1
                )[0]
                logging.warning(f"must use avoid interval {interval}")
                yield cur_pitch + interval
                avoid_intervals.remove(interval)

        # -------------------------------------------------------------------------------
        # Try direct intervals
        # -------------------------------------------------------------------------------
        while direct_intervals:
            interval = self._ic.get_interval_indices(
                direct_intervals, direct_harmonic_intervals, n=1
            )[0]
            logging.warning(f"must use direct interval {interval}")
            yield cur_pitch + interval
            direct_intervals.remove(interval)

    def _get_next_melody_pitch(
        self, score: _ScoreBase, i: int, next_bass_pitch: Pitch | None
    ) -> t.Iterator[Pitch]:
        cur_bass_pitch = score.structural_bass[i - 1]
        next_bass_pc = score.pc_bass[i]
        next_chord = score.chords[i]
        cur_melody_pitch = score.structural_melody[i - 1]
        dont_double_bass_pc = (
            next_chord.get_pitch_tendency(next_bass_pc) is not Tendency.NONE
        )
        # "no suspension" is understood as `suspension is None`
        for suspension in self._choose_suspension(
            score,
            next_chord,
            cur_melody_pitch,
            {next_bass_pc} if dont_double_bass_pc else set(),
        ):
            self._lingering_tendencies.append({})
            if suspension is not None:
                yield self._apply_suspension(score, i, *suspension)
                # we only continue execution if the recursive calls fail
                #   somewhere further down the stack, in which case we need to
                #   undo the suspension.
                self._undo_suspension(score, i)
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if next_bass_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        cur_melody_pitch,
                        [(cur_bass_pitch, next_bass_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.MELODY, pitch_in_other_voice=next_bass_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    cur_melody_pitch,
                    next_chord.pcs,
                    min_pitch,
                    max_pitch,
                    max_interval=self.settings.max_interval,
                    forbidden_intervals=forbidden_intervals,
                    allow_steps_outside_of_range=self.settings.allow_steps_outside_of_range,
                )

                # TODO for now tendency tones are only considered if there is
                #   no suspension.

                # First, try to proceed according to the tendency of the current
                # pitch
                for pitch in self._get_tendency(
                    score,
                    i,
                    cur_melody_pitch,
                    next_chord,
                    intervals,
                    dont_double_bass_pc,
                ):
                    # self._get_tendency removes intervals
                    logging.debug(
                        f"{self.__class__.__name__} yielding tendency resolution pitch {pitch}"
                    )

                    yield pitch
                    # self._lingering_tendencies[i + 1][
                    #     pitch
                    # ] = self.settings.tendency_decay_per_measure
                    # TODO: (Malcolm 2023-07-16) somewhere we need to pop self._lingering_tendencies

                # If the current tone does not have a tendency, or proceeding
                # according to the tendency doesn't work, try the other intervals
                yield from self._choose_intervals(
                    cur_bass_pitch,
                    next_bass_pitch,
                    cur_melody_pitch,
                    dont_double_bass_pc,
                    intervals,
                    voice_to_choose_for=OuterVoice.MELODY,
                )
            # self._lingering_tendencies.pop()

    def _get_next_bass_pitch(
        self, score: _ScoreBase, i: int, next_melody_pitch: Pitch | None
    ) -> t.Iterator[Pitch]:
        next_chord = score.chords[i]
        cur_bass_pitch = score.structural_bass[i - 1]
        cur_melody_pitch = score.structural_melody[i - 1]

        if next_melody_pitch is None:
            dont_double_melody_pc = False
        elif i in score.melody_suspensions:
            # Don't double a suspension
            dont_double_melody_pc = True
        else:
            # Don't double the pc if it is a tendency tone
            dont_double_melody_pc = (
                next_chord.get_pitch_tendency(next_melody_pitch) is not Tendency.NONE
            )

        for suspension in [None]:
            if suspension is not None:
                raise NotImplementedError("# TODO: (Malcolm 2023-07-15) ")
            else:
                # no suspension (suspension is None)

                # Find the intervals that would create forbidden parallels/antiparallels
                # so they can be excluded below
                if next_melody_pitch is None:
                    forbidden_intervals = []
                else:
                    forbidden_intervals = get_forbidden_intervals(
                        cur_bass_pitch,
                        [(cur_melody_pitch, next_melody_pitch)],
                        self.settings.forbidden_parallels,
                        self.settings.forbidden_antiparallels,
                    )

                # Get pitch bounds
                min_pitch, max_pitch = self._get_boundary_pitches(
                    OuterVoice.BASS, pitch_in_other_voice=next_melody_pitch
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    cur_bass_pitch,
                    (next_chord.foot,),
                    min_pitch,
                    max_pitch,
                    max_interval=self.settings.max_interval,
                    forbidden_intervals=forbidden_intervals,
                    allow_steps_outside_of_range=self.settings.allow_steps_outside_of_range,
                )

                # Compared with the melody implementation, we don't worry about tendency
                # tones since the next PC is determined by the chord symbol in any case

                yield from self._choose_intervals(
                    cur_melody_pitch,
                    next_melody_pitch,
                    cur_bass_pitch,
                    dont_double_melody_pc,
                    intervals,
                    voice_to_choose_for=OuterVoice.MELODY,
                )

    def _melody_step(
        self, score: _ScoreBase, bass_pitch: Pitch | None = None
    ) -> t.Iterator[Pitch]:
        i = len(score.structural_melody)

        # -------------------------------------------------------------------------------
        # Condition 1: start the melody
        # -------------------------------------------------------------------------------

        if not i:
            # generate the first note
            for pitch in self._get_first_melody_pitch(score, i, bass_pitch):
                logging.debug(f"{self.__class__.__name__} yielding first pitch {pitch}")
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        elif i in self._suspension_resolutions:
            for pitch in self._resolve_suspension(i):
                logging.debug(
                    f"{self.__class__.__name__} yielding suspension resolution pitch {pitch}"
                )
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            yield from self._get_next_melody_pitch(score, i, bass_pitch)

    def _bass_step(
        self, score: _ScoreBase, melody_pitch: Pitch | None = None
    ) -> t.Iterator[Pitch]:
        i = len(score.structural_bass)

        # -------------------------------------------------------------------------------
        # Condition 1: start the bass
        # -------------------------------------------------------------------------------

        if not i:
            for pitch in self._get_first_bass_pitch(score, i, melody_pitch):
                logging.debug(f"{self.__class__.__name__} yielding first pitch {pitch}")
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        # TODO: (Malcolm 2023-07-15)

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            for pitch in self._get_next_bass_pitch(score, i, melody_pitch):
                yield pitch

    def _step(self, score: _ScoreBase) -> t.Iterator[t.Dict[str, Pitch]]:
        if self.settings.do_first is OuterVoice.BASS:
            for bass_pitch in self._bass_step(score):
                for melody_pitch in self._melody_step(score, bass_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}
        else:
            for melody_pitch in self._melody_step(score):
                for bass_pitch in self._bass_step(score, melody_pitch):
                    yield {"bass": bass_pitch, "melody": melody_pitch}

        raise DeadEnd()

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[str] = None,
        range_constraints: RangeConstraints | None = None,
    ):
        """
        Args:
            chord_data: if string, should be in roman-text format.
                If a list, should be the output of the get_chords_from_rntxt
                function or similar.
        """
        if range_constraints is None:
            range_constraints = self.settings.range_constraints

        score = Score(chord_data, range_constraints=range_constraints)
        while True:
            if len(score.structural_melody) == len(score.chords):
                break
            next_pitches = next(self._step(score))
            # TODO: (Malcolm 2023-07-17) what about recursion here?
            score.structural_melody.append(next_pitches["melody"])
            score.structural_bass.append(next_pitches["bass"])
        return score

    def get_mididf_from_score(self, score: _ScoreBase):
        out_dict = {
            "onset": [chord.onset for chord in score.chords],
            "release": [chord.release for chord in score.chords],
            "bass": score.structural_bass,
            "melody": score.structural_melody,
        }
        out_df = pd.DataFrame(out_dict)
        return homodf_to_mididf(out_df)
