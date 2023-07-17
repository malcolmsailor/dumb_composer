"""Provides a class, IntervalChooser, that chooses small intervals more often than 
larger intervals."""
import math
import random
import typing as t
from collections import Counter  # used by doctests
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

NonnegativeFloat = float

# TODO: (Malcolm 2023-07-14) update so that we can also weight intervals arbitrarily


@dataclass
class IntervalChooserSettings:
    """
    lambda_: float >= 0; like the parameter to an exponential distribution.
        However, the intervals aren't drawn from a real exponential
        distribution because:
            - the distribution is discrete.
            - only the selected intervals can be chosen.
            - both positive and negative values are possible. (The weights
                of the negative values are chosen according to their
                absolute value.)
        Nevertheless the exponential curve seems to have an appropriate
        shape. If lambda_ = 0, then all intervals are equally likely.
        As it increases, small intervals become more likely.
    unison_weighted_as: In many contexts, we want melodic unisons to be
        relatively rare; in this case, we can set "unison_weighted_as" to a
        relatively high value (e.g., 3).
    """

    lambda_: NonnegativeFloat = 0.25
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

    Increasing lambda concentrates the distribution on smaller intervals:
    >>> ic = IntervalChooser(IntervalChooserSettings(lambda_=1.0))
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({0: 775, 2: 109, -2: 103, 5: 6, -5: 4, 7: 2, -7: 1})

    Zero lambda (uniform distribution):
    >>> ic = IntervalChooser(IntervalChooserSettings(lambda_=0.0))
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-7: 156, 7: 153, -5: 150, 0: 146, 5: 135, 2: 132, -2: 128})

    Reweighting unison:
    >>> ic = IntervalChooser(IntervalChooserSettings(lambda_=1.0, unison_weighted_as=7))
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-2: 473, 2: 468, 5: 33, -5: 20, 7: 2, -7: 2, 0: 2})
    """

    def __init__(self, settings: t.Optional[IntervalChooserSettings] = None):
        if settings is None:
            settings = IntervalChooserSettings()
        self._weights_memo = {}
        self._lambda = settings.lambda_
        self._unison_weighted_as = settings.unison_weighted_as

    def plot_weights(self, lower_bound=-10, upper_bound=10, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x = np.arange(lower_bound, upper_bound + 1)
        ax.plot(
            x,
            [self._get_weight(i) for i in x],
            label=f"lambda = {self._lambda}",
        )
        return ax

    def _get_weight(self, interval):
        return self._lambda * math.exp(
            -self._lambda
            * (self._unison_weighted_as if interval == 0 else abs(interval))
        )

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

        if intervals in self._weights_memo:
            interval_weights = self._weights_memo[intervals]
        else:
            interval_weights = [self._get_weight(interval) for interval in intervals]
            weight_sum = sum(interval_weights)
            if weight_sum == 0:
                interval_weights = [1.0 / len(intervals) for _ in intervals]
            else:
                interval_weights = [x / weight_sum for x in interval_weights]
            self._weights_memo[intervals] = interval_weights

        if custom_weights is None:
            weights = interval_weights
        else:
            weights = [x + y for x, y in zip(custom_weights, interval_weights)]

        out = random.choices(intervals, weights=weights, k=n)

        return out

    def __call__(
        self,
        intervals: t.Sequence[int],
        custom_weights: t.Sequence[float] | None = None,
    ) -> int:
        return self.choose_intervals(intervals, n=1, custom_weights=custom_weights)[0]


# if __name__ == "__main__":
#     lambdas = [2**i for i in range(-5, 2)]
#     fig, ax = plt.subplots()
#     for lda in lambdas:
#         ic = IntervalChooser(lambda_=lda)
#         ic.plot_weights(ax=ax)
#     plt.legend()
#     plt.show()
