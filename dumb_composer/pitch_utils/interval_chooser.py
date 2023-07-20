"""Provides a class, IntervalChooser, that chooses small intervals more often than 
larger intervals."""
import math
import random
import typing as t
from collections import Counter  # used by doctests
from dataclasses import dataclass, field
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np

from dumb_composer.constants import (
    DISSONANCE_WEIGHT,
    IMPERFECT_CONSONANCE_WEIGHT,
    OCTAVE_UNISON_WEIGHT,
    PERFECT_CONSONANCE_WEIGHT,
    TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS,
)
from dumb_composer.pitch_utils.intervals import reduce_compound_interval
from dumb_composer.pitch_utils.types import ChromaticInterval, Weight
from dumb_composer.utils.math_ import softmax

NonnegativeFloat = float

# TODO: (Malcolm 2023-07-14) update so that we can also weight intervals arbitrarily


@dataclass
class IntervalChooserSettings:
    """
    smaller_mel_interval_concentration: float >= 0; like the lambda parameter to an
        exponential distribution. However, the intervals aren't drawn from a real
        exponential distribution because:
            - the distribution is discrete.
            - only the selected intervals can be chosen.
            - both positive and negative values are possible. (The weights
                of the negative values are chosen according to their
                absolute value.)
        Nevertheless the exponential curve seems to have an appropriate
        shape. If smaller_mel_interval_concentration = 0, then all intervals are equally likely.
        As it increases, small intervals become more likely.
    unison_weighted_as: In many contexts, we want melodic unisons to be
        relatively rare; in this case, we can set "unison_weighted_as" to a
        relatively high value (e.g., 3).
    """

    smaller_mel_interval_concentration: NonnegativeFloat = 1.25
    unison_weighted_as: int = 0


class IntervalChooser:
    """

    ------------------------------------------------------------------------------------
    Default settings
    ------------------------------------------------------------------------------------

    >>> ic = IntervalChooser()
    >>> intervals = [-7, -5, -2, 0, 2, 5, 7]

    To get a single interval, call an instance directly
    >>> ic(intervals)  # doctest: +SKIP
    0

    To get a list of intervals use `choose_intervals()`
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({0: 323, -2: 190, 2: 189, -5: 97, 5: 93, -7: 57, 7: 51})

    ------------------------------------------------------------------------------------
    Custom settings
    ------------------------------------------------------------------------------------

    Increasing smaller_mel_interval_concentration concentrates the distribution on
    smaller intervals:
    >>> ic = IntervalChooser(
    ...     IntervalChooserSettings(smaller_mel_interval_concentration=1.0)
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({0: 775, 2: 109, -2: 103, 5: 6, -5: 4, 7: 2, -7: 1})

    Zero smaller_mel_interval_concentration (uniform distribution):
    >>> ic = IntervalChooser(
    ...     IntervalChooserSettings(smaller_mel_interval_concentration=0.0)
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-7: 156, 7: 153, -5: 150, 0: 146, 5: 135, 2: 132, -2: 128})

    Reweighting unison:
    >>> ic = IntervalChooser(
    ...     IntervalChooserSettings(
    ...         smaller_mel_interval_concentration=1.0, unison_weighted_as=7
    ...     )
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-2: 473, 2: 468, 5: 33, -5: 20, 7: 2, -7: 2, 0: 2})
    """

    def __init__(self, settings: t.Optional[IntervalChooserSettings] = None):
        if settings is None:
            settings = IntervalChooserSettings()
        self._weights_memo = {}
        self._smaller_mel_interval_concentration = (
            settings.smaller_mel_interval_concentration
        )
        self._unison_weighted_as = settings.unison_weighted_as

    def plot_weights(self, lower_bound=-10, upper_bound=10, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x = np.arange(lower_bound, upper_bound + 1)
        ax.plot(
            x,
            [self._get_melodic_interval_weight(i) for i in x],
            label=f"lambda = {self._smaller_mel_interval_concentration}",
        )
        return ax

    def _get_melodic_interval_weight(self, interval):
        # Exponential distribution is lambda * exp(-lambda * x)
        # However, scaling by lambda is just done to normalize the distribution
        # so it integrates to 1. For a number of reasons, this isn't necessary here.
        return math.exp(
            -self._smaller_mel_interval_concentration
            * (self._unison_weighted_as if interval == 0 else abs(interval))
        )

    def _get_melodic_interval_weights(self, intervals: t.Tuple[int]):
        if intervals in self._weights_memo:
            interval_weights = self._weights_memo[intervals]
        else:
            interval_weights = [
                self._get_melodic_interval_weight(interval) for interval in intervals
            ]
            weight_sum = sum(interval_weights)
            if weight_sum == 0:
                interval_weights = [1.0 / len(intervals) for _ in intervals]
            else:
                interval_weights = [x / weight_sum for x in interval_weights]
            self._weights_memo[intervals] = interval_weights
        return interval_weights

    @staticmethod
    def _make_weighted_choice(
        intervals, *weights: t.Sequence[Weight] | None, n: int = 1
    ):
        filtered_weights = [w for w in weights if w is not None]
        if len(filtered_weights) > 1:
            consolidated_weights = [sum(ws) for ws in zip(*filtered_weights)]
        else:
            consolidated_weights = filtered_weights[0]

        return random.choices(intervals, weights=consolidated_weights, k=n)

    def choose_intervals(
        self,
        intervals: t.Sequence[int],
        n: int = 1,
        custom_weights: t.Sequence[float] | None = None,
    ) -> t.List[int]:
        """
        For now we are just summing `custom_weights` with `interval_weights`.
        The latter are normalized so they always sum to 1.
        """

        intervals = tuple(intervals)

        interval_weights = self._get_melodic_interval_weights(intervals)

        return self._make_weighted_choice(
            intervals, interval_weights, custom_weights, n=n
        )

    def __call__(
        self,
        intervals: t.Sequence[int],
        custom_weights: t.Sequence[float] | None = None,
    ) -> int:
        return self.choose_intervals(intervals, n=1, custom_weights=custom_weights)[0]


@dataclass
class HarmonicallyInformedIntervalChooserSettings(IntervalChooserSettings):
    harmonic_interval_weights: t.Mapping[ChromaticInterval, Weight] = field(
        default_factory=lambda: TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS.copy()
    )


class HarmonicallyInformedIntervalChooser(IntervalChooser):
    def __init__(
        self, settings: HarmonicallyInformedIntervalChooserSettings | None = None
    ):
        if settings is None:
            settings = HarmonicallyInformedIntervalChooserSettings()
        super().__init__(settings)
        softmaxed_weights = softmax(list(settings.harmonic_interval_weights.values()))
        self._harmonic_interval_weights = {
            k: v
            for k, v in zip(
                settings.harmonic_interval_weights.keys(), softmaxed_weights
            )
        }

    def get_interval_indices(
        self,
        melodic_intervals: t.Sequence[int],
        harmonic_intervals: t.Sequence[int] | None,
        n: int = 1,
        custom_weights: t.Sequence[float] | None = None,
    ) -> t.List[int]:
        melodic_intervals = tuple(melodic_intervals)
        melodic_interval_weights = self._get_melodic_interval_weights(melodic_intervals)
        if harmonic_intervals is None:
            harmonic_interval_weights = None
        else:
            harmonic_interval_weights = [
                self._harmonic_interval_weights[h % 12] for h in harmonic_intervals
            ]

        return self._make_weighted_choice(
            list(range(len(melodic_intervals))),
            melodic_interval_weights,
            harmonic_interval_weights,
            custom_weights,
            n=n,
        )

    def choose_intervals(
        self,
        melodic_intervals: t.Sequence[int],
        harmonic_intervals: t.Sequence[int] | None,
        n: int = 1,
        custom_weights: t.Sequence[float] | None = None,
    ) -> t.List[int]:
        indices = self.get_interval_indices(
            melodic_intervals, harmonic_intervals, n, custom_weights
        )
        return [melodic_intervals[i] for i in indices]

    def __call__(
        self,
        melodic_intervals: t.Sequence[int],
        harmonic_intervals: t.Sequence[int] | None,
        custom_weights: t.Sequence[float] | None = None,
    ) -> int:
        return self.choose_intervals(
            melodic_intervals,
            harmonic_intervals=harmonic_intervals,
            n=1,
            custom_weights=custom_weights,
        )[0]


# if __name__ == "__main__":
#     lambdas = [2**i for i in range(-5, 2)]
#     fig, ax = plt.subplots()
#     for lda in lambdas:
#         ic = IntervalChooser(smaller_mel_interval_concentration=lda)
#         ic.plot_weights(ax=ax)
#     plt.legend()
#     plt.show()
