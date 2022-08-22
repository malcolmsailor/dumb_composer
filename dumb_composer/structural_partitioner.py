from dataclasses import dataclass
from functools import cached_property
import math
from numbers import Number
import random
import typing as t
import numpy as np
import pandas as pd


from .time import Meter

from .time_utils import (
    get_barline_times_within_duration,
    get_onset_closest_to_middle_of_duration,
)
from .shared_classes import Score


@dataclass
class StructuralPartitionerSettings:
    never_split_dur_in_beats: float = 1.0
    always_split_dur_in_bars: float = 2.0

    # If we are splitting a chord that does not overlap with a barline, we get
    # the metric weights and than apply a softmax to turn this into a
    # probability distribution we can sample from. The higher the temperature,
    # the more the probability mass is concentrated at the higher metric
    # weights. (The temperature should always be > 0.)
    candidates_softmax_temperature: float = 1.0


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()


def _flatten_list_sub(x: t.Union[t.Any, t.List[t.Any]]):
    if isinstance(x, list):
        out = []
        for y in x:
            out += _flatten_list_sub(y)
        return out
    else:
        return [x]


def flatten_list(l: t.List[t.Any]):
    """
    >>> flatten_list([1, 2, 3])
    [1, 2, 3]
    >>> flatten_list([])
    []
    >>> flatten_list([1, [2, 3], [[4], [], [[5, 6]]]])
    [1, 2, 3, 4, 5, 6]
    """
    return _flatten_list_sub(l)


class StructuralPartitioner:
    def __init__(
        self, settings: t.Optional[StructuralPartitionerSettings] = None
    ):
        if settings is None:
            settings = StructuralPartitionerSettings()
        self.settings = settings
        self._ts = None

    @cached_property
    def _linear(self):
        x1 = self.settings.never_split_dur_in_beats * self._ts.beat_dur
        slope = 1 / (
            self.settings.always_split_dur_in_bars * self._ts.bar_dur - x1
        )
        return lambda x: slope * (x - x1)

    def _step(
        self, chord_onset: Number, chord_release: Number
    ) -> t.Union[t.Tuple[Number, Number], t.List]:
        # TODO for now, we just split once. That means a two-bar chord is
        # guaranteed to be split into two one-bar chords, but these
        # one-bar chords will *not* be split further. Whereas each original
        # one-bar chord has a ~50% chance of being split. That seems wrong.
        chord_dur = chord_release - chord_onset
        split = random.random() < self._linear(chord_dur)
        if not split:
            return (chord_onset, chord_release)

        barline_times = get_barline_times_within_duration(
            chord_onset,
            chord_release,
            self._ts.bar_dur,
            include_start=False,
        )
        if barline_times:
            # if we are splitting the chord and it overlaps with at least one
            #   barline, then we split at the barline closest to the middle of the
            #   chord.
            split_point = get_onset_closest_to_middle_of_duration(
                barline_times, chord_onset, chord_release
            )
        else:
            # Otherwise, we choose a split point with probability proportional to
            #   the metric strength of each point in the bar, down to
            #   never_split_dur_in_beats
            candidates = self._ts.weights_between(
                math.ceil(self.settings.never_split_dur_in_beats)
                * self._ts.beat_dur,
                chord_onset,
                chord_release,
                include_start=False,
                out_format="dict",
            )
            candidate_onsets, candidate_weights = zip(*candidates.items())
            probs = softmax(candidate_weights)
            split_point = random.choices(candidate_onsets, weights=probs)[0]
        return [
            self._step(chord_onset, split_point),
            self._step(split_point, chord_release),
        ]

    def __call__(self, score: Score):
        """Updates score in place.

        TODO it would be good to preserve the original "score.chords" somehow.
        """
        self._ts = Meter(score.ts)
        split_chords = []
        for _, chord in score.chords.iterrows():
            splits = self._step(chord.onset, chord.release)
            for start, stop in flatten_list(splits):
                new_chord = chord.copy()
                new_chord.onset = start
                new_chord.release = stop
                split_chords.append(new_chord)
        score.chords = pd.DataFrame(split_chords).reset_index(drop=True)


if __name__ == "__main__":
    settings = StructuralPartitionerSettings(ts="3/4")
    sp = StructuralPartitioner(settings)
    for x in range(0, 100):
        print(x * 0.1, sp._linear(x * 0.1))
