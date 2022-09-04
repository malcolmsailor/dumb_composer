from functools import cached_property
from numbers import Number
import random
import numpy as np
import pandas as pd


class RhythmistBase:
    pass


class RuleBasedRhythmist(RhythmistBase):
    def __init__(self, ts, min_subdivision):
        pass

    @cached_property
    def onset_positions(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class DumbRhythmist(RhythmistBase):
    """Temporary class until I implement an ML rhythm generating model."""

    def __init__(
        self,
        # n_measures: int,
        measure_dur: Number,  # TODO infer from time sig
        min_dur: Number = 0.25,
        tactus: Number = 1.0,
        max_weight: int = 2,
        min_weight: int = -2,
        max_prob: float = 0.9,
        min_prob: float = 0.2,
    ):
        self._tactus = tactus
        self._max_weight = max_weight
        self._min_weight = min_weight
        self._max_prob = max_prob
        self._min_prob = min_prob
        # self._n_measures = n_measures
        self._measure_dur = measure_dur
        assert measure_dur % min_dur == 0
        self._min_dur = min_dur
        self._prob = self._get_prob()

    def _get_prob(self):
        x1 = self._min_weight
        x2 = self._max_weight
        y1 = self._min_prob
        y2 = self._max_prob
        m = (y2 - y1) / (x2 - x1)
        return lambda x: m * (x - x1) + y1

    # @property
    # def _total_dur(self):
    #     return self._n_measures * self._measure_dur

    def _get_weight(self, x):
        for weight in range(self._max_weight, self._min_weight, -1):
            if not x % (2**weight):
                break
        return weight

    def __call__(self, n_measures: int, end_with_downbeat: bool = True):
        if end_with_downbeat:
            assert n_measures >= 1
            n_measures -= 1
        total_dur = n_measures * self._measure_dur
        onset_positions = np.arange(0, total_dur, self._min_dur)
        onset_weights = [self._get_weight(x) for x in onset_positions]
        onsets = []
        for position, weight in zip(onset_positions, onset_weights):
            prob = self._prob(weight)
            if random.random() < prob:
                onsets.append(position)
        if end_with_downbeat:
            onsets.append(total_dur)
            total_dur += self._measure_dur
        # for now, releases just go until next onset
        releases = onsets[1:]
        releases.append(total_dur)

        return pd.DataFrame({"onset": onsets, "release": releases})
