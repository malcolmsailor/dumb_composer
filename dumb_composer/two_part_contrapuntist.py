import logging
import random
import sys
import typing as t
from dataclasses import dataclass
from numbers import Number

import pandas as pd

from dumb_composer.constants import DEFAULT_BASS_RANGE, DEFAULT_MEL_RANGE
from dumb_composer.from_ml_out import get_chord_df
from dumb_composer.pitch_utils.chords import Chord, Tendency, get_chords_from_rntxt
from dumb_composer.pitch_utils.interval_chooser import (
    IntervalChooser,
    IntervalChooserSettings,
)
from dumb_composer.pitch_utils.intervals import get_forbidden_intervals, interval_finder
from dumb_composer.pitch_utils.put_in_range import get_all_in_range, put_in_range
from dumb_composer.pitch_utils.spacings import RangeConstraints
from dumb_composer.pitch_utils.types import ChromaticInterval, Pitch, TimeStamp
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf

from .shared_classes import Annotation, Score, _ScoreBase
from .suspensions import Suspension, find_suspension_release_times, find_suspensions
from .utils.math_ import softmax, weighted_sample_wo_replacement
from .utils.recursion import DeadEnd

# TODO bass suspensions

# TODO don't permit suspension resolutions to tendency tones (I think I may have
# done this, double check)


@dataclass
class TwoPartContrapuntistSettings(IntervalChooserSettings):
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
    no_suspension_score: float = 1.0
    allow_avoid_intervals: bool = False
    allow_steps_outside_of_range: bool = True
    range_constraints: RangeConstraints = RangeConstraints()
    expected_total_number_of_voices: int = 4

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type:ignore


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
        self._ic = IntervalChooser(settings)
        self._suspension_resolutions: t.Dict[int, int] = {}
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
        cur_mel_pitch: int,
        next_bass_has_tendency: bool,
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
        if next_bass_has_tendency:
            eligible_pcs = next_chord.pcs[1:]
        else:
            eligible_pcs = next_chord.pcs
        suspensions = find_suspensions(
            # TODO providing next_scale_pcs prohibits "chromatic" suspensions
            #   (see the definition of find_suspensions()) but in the long
            #   run I'm not sure whether this is indeed something we want to
            #   do.
            cur_mel_pitch,
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
        cur_mel_pitch: int,
        next_chord: Chord,
        intervals: t.List[int],
        next_bass_has_tendency: bool,
    ) -> t.Iterable[int]:
        cur_chord = score.chords[i - 1]
        tendency = cur_chord.get_pitch_tendency(cur_mel_pitch)
        if tendency is Tendency.NONE or cur_chord == next_chord:
            return
        if tendency is Tendency.UP:
            steps = (1, 2)
        else:
            steps = (-1, -2)
        for step in steps:
            if step in intervals:
                candidate_pitch = cur_mel_pitch + step
                if next_bass_has_tendency and (
                    candidate_pitch % 12 == score.structural_bass[i] % 12
                ):
                    return
                yield candidate_pitch
                intervals.remove(step)
                return
        logging.debug(
            f"resolving tendency-tone with pitch {cur_mel_pitch} " f"would exceed range"
        )

    def _get_first_melody_pitch(self, score: _ScoreBase, i: int):
        next_bass_pitch = score.structural_bass[i]
        next_chord = score.chords[i]
        eligible_pcs = next_chord.get_pcs_that_can_be_added_to_existing_voicing()

        # TODO: (Malcolm 2023-07-14) weight choices according to interval
        mel_pitch_choices = get_all_in_range(
            eligible_pcs,
            max(next_bass_pitch, score.range_constraints.min_melody_pitch),
            score.range_constraints.max_melody_pitch,
        )
        random.shuffle(mel_pitch_choices)
        yield from mel_pitch_choices

    def _resolve_suspension(self, i: int):
        yield self._suspension_resolutions[i]

    def _choose_intervals(
        self,
        cur_bass_pitch: Pitch,
        next_bass_pitch: Pitch,
        next_bass_has_tendency: bool,
        cur_mel_pitch: Pitch,
        intervals: t.List[ChromaticInterval],
    ):
        avoid_intervals = []
        direct_intervals = []
        while intervals:
            interval = self._ic(intervals)
            candidate_pitch = cur_mel_pitch + interval

            # Check for avoid intervals

            if next_bass_has_tendency and (candidate_pitch % 12) == (
                next_bass_pitch % 12
            ):
                # Don't double tendency tones in bass
                intervals.remove(interval)
                avoid_intervals.append(interval)
                continue

            bass_interval = next_bass_pitch - cur_bass_pitch

            # TODO: (Malcolm 2023-07-14) As far as I can tell `accomp_bass_range` no
            #   longer exists. I'm commenting out this code but leaving it for the
            #   time being to see if there are any associated issues.
            # adjusting the bass_intervals in this way isn't really ideal
            #   I think it is only necessary because "bass_range" and
            #   "accomp_bass_range" don't necessarily agree. I should
            #   enforce agreement between them.
            # if abs(bass_interval) > 6:
            #     if bass_interval > 0:
            #         bass_interval -= 12
            #     else:
            #         bass_interval += 12

            # Check for direct intervals

            if (
                abs(interval) > 2
                and (
                    (candidate_pitch - next_bass_pitch) % 12
                    in self.settings.unpreferred_direct_intervals
                )
                and (
                    (0 not in (interval, bass_interval))
                    and ((interval > 0) == (bass_interval > 0))
                )
            ):
                # We move unpreferred direct intervals into another list and only
                # consider them later if we need to
                intervals.remove(interval)
                direct_intervals.append(interval)
                continue

            # Try interval

            yield candidate_pitch
            intervals.remove(interval)

        # -------------------------------------------------------------------------------
        # Try avoid intervals
        # -------------------------------------------------------------------------------
        if self.settings.allow_avoid_intervals:
            while avoid_intervals:
                interval = self._ic(avoid_intervals)
                logging.warning(f"must use avoid interval {interval}")
                yield cur_mel_pitch + interval
                avoid_intervals.remove(interval)

        # -------------------------------------------------------------------------------
        # Try direct intervals
        # -------------------------------------------------------------------------------
        while direct_intervals:
            interval = self._ic(direct_intervals)
            logging.warning(f"must use direct interval {interval}")
            yield cur_mel_pitch + interval
            direct_intervals.remove(interval)

    def _get_next_melody_pitch(self, score: _ScoreBase, i: int):
        cur_bass_pitch = score.structural_bass[i - 1]
        next_bass_pitch = score.structural_bass[i]
        next_chord = score.chords[i]
        cur_mel_pitch = score.structural_melody[i - 1]
        next_bass_has_tendency = (
            next_chord.get_pitch_tendency(next_bass_pitch) is not Tendency.NONE
        )
        # "no suspension" is understood as `suspension is None`
        for suspension in self._choose_suspension(
            score, next_chord, cur_mel_pitch, next_bass_has_tendency
        ):
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
                forbidden_intervals = get_forbidden_intervals(
                    cur_mel_pitch,
                    [(cur_bass_pitch, next_bass_pitch)],
                    self.settings.forbidden_parallels,
                    self.settings.forbidden_antiparallels,
                )

                # Get a list of all available intervals
                intervals = interval_finder(
                    cur_mel_pitch,
                    next_chord.pcs,
                    score.range_constraints.min_melody_pitch,
                    score.range_constraints.max_melody_pitch,
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
                    cur_mel_pitch,
                    next_chord,
                    intervals,
                    next_bass_has_tendency,
                ):
                    # self._get_tendency removes intervals
                    logging.debug(
                        f"{self.__class__.__name__} yielding tendency resolution pitch {pitch}"
                    )
                    yield pitch

                # If the current tone does not have a tendency, or proceeding
                # according to the tendency doesn't work, try the other intervals
                for pitch in self._choose_intervals(
                    cur_bass_pitch,
                    next_bass_pitch,
                    next_bass_has_tendency,
                    cur_mel_pitch,
                    intervals,
                ):
                    logging.debug(f"{self.__class__.__name__} yielding pitch {pitch}")
                    yield pitch

    def _step(self, score: _ScoreBase):
        i = len(score.structural_melody)

        # -------------------------------------------------------------------------------
        # Condition 1: start the melody
        # -------------------------------------------------------------------------------

        if not i:
            # generate the first note
            for pitch in self._get_first_melody_pitch(score, i):
                logging.debug(f"{self.__class__.__name__} yielding first pitch {pitch}")
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 2: there is an ongoing suspension to resolve
        # -------------------------------------------------------------------------------

        elif i in self._suspension_resolutions:
            # TODO: (Malcolm 2023-07-14) I think we should check if
            for pitch in self._resolve_suspension(i):
                logging.debug(
                    f"{self.__class__.__name__} yielding suspension resolution pitch {pitch}"
                )
                yield pitch

        # -------------------------------------------------------------------------------
        # Condition 3: choose a note freely
        # -------------------------------------------------------------------------------

        else:
            for pitch in self._get_next_melody_pitch(score, i):
                yield pitch

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
            next_pitch = next(self._step(score))
            score.structural_melody.append(next_pitch)
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
