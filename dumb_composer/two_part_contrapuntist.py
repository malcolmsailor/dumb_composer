from dataclasses import dataclass
import random

import typing as t

import pandas as pd

from dumb_composer.pitch_utils.chords import get_chords_from_rntxt, Chord
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
from .suspensions import find_suspensions
from .utils.recursion import DeadEnd
from .shared_classes import Score
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf
from dumb_composer.from_ml_out import get_chord_df


@dataclass
class TwoPartContrapuntistSettings(IntervalChooserSettings):
    bass_range: t.Tuple[int, int] = (30, 50)
    mel_range: t.Tuple[int, int] = (60, 78)
    forbidden_parallels: t.Sequence[int] = (7, 0)
    forbidden_antiparallels: t.Sequence[int] = (0,)
    max_interval: int = 12


class TwoPartContrapuntist:
    def __init__(
        self,
        settings=None,
    ):
        if settings is None:
            settings = TwoPartContrapuntistSettings()
        self._settings = settings
        # TODO the size of lambda should depend on how long the chord is.
        #   If a chord lasts for a whole note it can move by virtually any
        #   amount. If a chord lasts for an eighth note it should move by
        #   a relatively small amount.
        self._ic = IntervalChooser(self._settings)
        self._forbidden_parallels = settings.forbidden_parallels
        self._forbidden_antiparallels = settings.forbidden_antiparallels
        self._max_interval = settings.max_interval
        self._bass_range = settings.bass_range
        self._mel_range = settings.mel_range

    def _get_ranges(self, bass_range, mel_range):
        if bass_range is None:
            if self._bass_range is None:
                raise ValueError
            bass_range = self._bass_range
        if mel_range is None:
            if self._mel_range is None:
                raise ValueError
            mel_range = self._mel_range
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

    def _step(
        self,
        score: Score,
    ):
        i = len(score.structural_melody)
        next_bass_pitch = score.structural_bass[i]
        next_chord_pcs = score.chords[i].pcs
        if not i:
            # generate the first note
            mel_pitch_choices = get_all_in_range(
                next_chord_pcs,
                max(next_bass_pitch, score.mel_range[0]),
                score.mel_range[1],
            )
            while mel_pitch_choices:
                mel_pitch_i = random.randrange(len(mel_pitch_choices))
                yield mel_pitch_choices[mel_pitch_i]
                mel_pitch_choices.pop(mel_pitch_i)
        else:
            cur_mel_pitch = score.structural_melody[i - 1]
            # suspensions = find_suspensions(cur_mel_pitch, next_chord_pcs)
            # TODO finish suspensions
            cur_bass_pitch = score.structural_bass[i - 1]
            forbidden_intervals = get_forbidden_intervals(
                cur_mel_pitch,
                [(cur_bass_pitch, next_bass_pitch)],
                self._forbidden_parallels,
                self._forbidden_antiparallels,
            )
            intervals = interval_finder(
                cur_mel_pitch,
                next_chord_pcs,
                *score.mel_range,
                max_interval=self._max_interval,
                forbidden_intervals=forbidden_intervals,
            )
            while intervals:
                interval = self._ic(intervals)
                yield cur_mel_pitch + interval
                intervals.remove(interval)
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
        for _ in range(len(score.chords)):
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
