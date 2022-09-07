from dataclasses import dataclass
from numbers import Number
import random
import logging

import typing as t

import pandas as pd

from dumb_composer.pitch_utils.chords import (
    Tendency,
    get_chords_from_rntxt,
    Chord,
)
from dumb_composer.pitch_utils.put_in_range import (
    get_all_in_range,
    put_in_range,
)
from dumb_composer.pitch_utils.intervals import (
    get_forbidden_intervals,
    interval_finder,
)
from dumb_composer.pitch_utils.interval_chooser import (
    IntervalChooser,
    IntervalChooserSettings,
)
from .suspensions import Suspension, find_suspension_releases, find_suspensions
from .utils.recursion import DeadEnd
from .utils.math_ import softmax, weighted_sample_wo_replacement
from .shared_classes import Annotation, Score
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf
from dumb_composer.from_ml_out import get_chord_df
from dumb_composer.constants import DEFAULT_BASS_RANGE, DEFAULT_MEL_RANGE

# TODO bass suspensions

# TODO don't permit suspension resolutions to tendency tones (I think I may have
# done this, double check)


@dataclass
class TwoPartContrapuntistSettings(IntervalChooserSettings):
    bass_range: t.Optional[t.Tuple[int, int]] = None
    mel_range: t.Optional[t.Tuple[int, int]] = None
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
    no_suspension_score: float = 1.0
    allow_avoid_intervals: bool = False
    allow_steps_outside_of_range: bool = True

    def __post_init__(self):
        logging.debug(f"running TwoPartContrapuntistSettings __post_init__()")
        if self.bass_range is None:
            self.bass_range = DEFAULT_BASS_RANGE
        if self.mel_range is None:
            self.mel_range = DEFAULT_MEL_RANGE
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


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
            self._upward_suspension_resolutions = (
                self.settings.allow_upward_suspensions
            )

    def _get_ranges(self, bass_range, mel_range):
        if bass_range is None:
            if self.settings.bass_range is None:
                raise ValueError
            bass_range = self.settings.bass_range
        if mel_range is None:
            if self.settings.mel_range is None:
                raise ValueError
            mel_range = self.settings.mel_range
        return bass_range, mel_range

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
        return self(
            chord_data, ts, bass_range, mel_range, initial_mel_chord_factor
        )

    def _choose_suspension(
        self,
        score: Score,
        next_chord: Chord,
        cur_mel_pitch: int,
        next_bass_has_tendency: bool,
    ):
        suspension_releases = find_suspension_releases(
            next_chord.onset,
            next_chord.release,
            score.ts,
            max_weight_diff=self.settings.max_suspension_weight_diff,
            max_suspension_dur=self.settings.max_suspension_dur,
        )
        if not suspension_releases:
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
        for suspension in weighted_sample_wo_replacement(
            suspensions + [None], weights
        ):
            if suspension is None:
                yield None
            else:
                # TODO suspension_releases weights
                suspension_release = random.choices(suspension_releases, k=1)[0]
                yield suspension, suspension_release

    def _apply_suspension(
        self,
        score: Score,
        i: int,
        suspension: Suspension,
        suspension_release: Number,
    ) -> int:
        assert i + 1 not in self._suspension_resolutions
        score.split_ith_chord_at(i, suspension_release)
        suspended_pitch = score.structural_melody[i - 1]
        self._suspension_resolutions[i + 1] = (
            suspended_pitch + suspension.resolves_by
        )
        assert i not in score.suspension_indices
        score.suspension_indices.add(i)
        if self.settings.annotate_suspensions:
            score.annotations["suspensions"].append(
                Annotation(score.chords[i].onset, "S")
            )
        return suspended_pitch

    def _undo_suspension(self, score: Score, i: int) -> None:
        assert i + 1 in self._suspension_resolutions
        score.merge_ith_chords(i)
        del self._suspension_resolutions[i + 1]
        score.suspension_indices.remove(i)  # raises KeyError if not present
        if self.settings.annotate_suspensions:
            popped_annotation = score.annotations["suspensions"].pop()
            assert popped_annotation.onset == score.chords[i].onset

    def _get_tendency(
        self,
        score: Score,
        i: int,
        cur_mel_pitch: int,
        next_chord: Chord,
        intervals: t.List[int],
        next_bass_has_tendency: bool,
    ) -> t.Optional[int]:
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
        # if tendency is Tendency.DOWN:
        #     for step in (-1, -2):
        #         if step in intervals:
        #             candidate_pitch = cur_mel_pitch + step
        #             if next_bass_has_tendency and (
        #                 candidate_pitch % 12 == score.structural_bass[i] % 12
        #             ):
        #                 return
        #             yield candidate_pitch
        #             intervals.remove(step)
        #             return
        logging.debug(
            f"resolving tendency-tone with pitch {cur_mel_pitch} "
            f"would exceed range"
        )

    def _get_first_melody_pitch(self, score: Score, i: int):
        next_bass_pitch = score.structural_bass[i]
        next_chord = score.chords[i]

        mel_pitch_choices = get_all_in_range(
            next_chord.pcs,
            max(next_bass_pitch, score.mel_range[0]),
            score.mel_range[1],
        )
        while mel_pitch_choices:
            mel_pitch_i = random.randrange(len(mel_pitch_choices))
            yield mel_pitch_choices[mel_pitch_i]
            mel_pitch_choices.pop(mel_pitch_i)

    def _resolve_suspension(self, i: int):
        yield self._suspension_resolutions[i]

    def _choose_intervals(
        self,
        cur_bass_pitch: int,
        next_bass_pitch: int,
        next_bass_has_tendency: bool,
        cur_mel_pitch: int,
        intervals: t.List[int],
    ):
        avoid_intervals = []
        direct_intervals = []
        while intervals:
            interval = self._ic(intervals)
            candidate_pitch = cur_mel_pitch + interval
            if next_bass_has_tendency and (candidate_pitch % 12) == (
                next_bass_pitch % 12
            ):
                intervals.remove(interval)
                avoid_intervals.append(interval)
                continue
            bass_interval = next_bass_pitch - cur_bass_pitch
            # adjusting the bass_intervals in this way isn't really ideal
            #   I think it is only necessary because "bass_range" and
            #   "accomp_bass_range" don't necessarily agree. I should
            #   enforce agreement between them. TODO
            if abs(bass_interval) > 6:
                if bass_interval > 0:
                    bass_interval -= 12
                else:
                    bass_interval += 12
            # if interval == -3 and candidate_pitch == 72:
            #     breakpoint()
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
                intervals.remove(interval)
                direct_intervals.append(interval)
                continue
            yield candidate_pitch
            intervals.remove(interval)
        if self.settings.allow_avoid_intervals:
            while avoid_intervals:
                interval = self._ic(avoid_intervals)
                logging.warning(f"must use avoid interval {interval}")
                yield cur_mel_pitch + interval
                avoid_intervals.remove(interval)
        while direct_intervals:
            interval = self._ic(direct_intervals)
            logging.warning(f"must use direct interval {interval}")
            yield cur_mel_pitch + interval
            direct_intervals.remove(interval)

    def _get_next_melody_pitch(self, score: Score, i: int):
        cur_bass_pitch = score.structural_bass[i - 1]
        next_bass_pitch = score.structural_bass[i]
        next_chord = score.chords[i]
        cur_mel_pitch = score.structural_melody[i - 1]
        next_bass_has_tendency = (
            next_chord.get_pitch_tendency(next_bass_pitch) is not Tendency.NONE
        )
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
                cur_bass_pitch = score.structural_bass[i - 1]
                forbidden_intervals = get_forbidden_intervals(
                    cur_mel_pitch,
                    [(cur_bass_pitch, next_bass_pitch)],
                    self.settings.forbidden_parallels,
                    self.settings.forbidden_antiparallels,
                )
                intervals = interval_finder(
                    cur_mel_pitch,
                    next_chord.pcs,
                    *score.mel_range,
                    max_interval=self.settings.max_interval,
                    forbidden_intervals=forbidden_intervals,
                    allow_steps_outside_of_range=self.settings.allow_steps_outside_of_range,
                )
                # TODO for now tendency tones are only considered if there is
                #   no suspension.
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
                for pitch in self._choose_intervals(
                    cur_bass_pitch,
                    next_bass_pitch,
                    next_bass_has_tendency,
                    cur_mel_pitch,
                    intervals,
                ):
                    logging.debug(
                        f"{self.__class__.__name__} yielding pitch {pitch}"
                    )
                    yield pitch
                # while intervals:
                #     interval = self._ic(intervals)
                #     yield cur_mel_pitch + interval
                #     intervals.remove(interval)

    def _step(self, score: Score):
        i = len(score.structural_melody)
        if not i:
            # generate the first note
            for pitch in self._get_first_melody_pitch(score, i):
                logging.debug(
                    f"{self.__class__.__name__} yielding pitch {pitch}"
                )
                yield pitch
        elif i in self._suspension_resolutions:
            for pitch in self._resolve_suspension(i):
                logging.debug(
                    f"{self.__class__.__name__} yielding pitch {pitch}"
                )
                yield pitch
        else:
            for pitch in self._get_next_melody_pitch(score, i):
                yield pitch
        raise DeadEnd()

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[str] = None,
        bass_range: t.Optional[t.Tuple[int, int]] = None,
        mel_range: t.Optional[t.Tuple[int, int]] = None,
    ):
        """
        Args:
            chord_data: if string, should be in roman-text format.
                If a list, should be the output of the get_chords_from_rntxt
                function or similar.
        """
        bass_range, mel_range = self._get_ranges(bass_range, mel_range)
        score = Score(chord_data, bass_range, mel_range)
        while True:
            if len(score.structural_melody) == len(score.chords):
                break
            next_pitch = next(self._step(score))
            score.structural_melody.append(next_pitch)
        return score

    def get_mididf_from_score(self, score: Score):
        out_dict = {
            "onset": [chord.onset for chord in score.chords],
            "release": [chord.release for chord in score.chords],
            "bass": score.structural_bass,
            "melody": score.structural_melody,
        }
        # out_df = score.chords[["onset", "release"]].copy()
        # out_df["bass"] = score.structural_bass
        # out_df["melody"] = score.structural_melody
        out_df = pd.DataFrame(out_dict)
        return homodf_to_mididf(out_df)
