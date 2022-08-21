import typing as t
from dataclasses import dataclass
import math
import random

import matplotlib.pyplot as plt
import numpy as np


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
        As it increases, small intervals become correspondingly more likely.
    unison_weighted_as: In many contexts, we want melodic unisons to be
        relatively rare; in this case, we can set "unison_weighted_as" to a
        relatively high value (e.g., 3).
    """

    lambda_: float = 0.25
    unison_weighted_as: int = 0


class IntervalChooser:
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

    def __call__(self, intervals, n=1):
        """Intervals should ideally be sorted so that the memo-ing is accurate
        but it probably isn't worth actually sorting them.
        """
        intervals = tuple(intervals)
        if intervals in self._weights_memo:
            weights = self._weights_memo[intervals]
        else:
            weights = [self._get_weight(interval) for interval in intervals]
            if sum(weights) == 0:
                weights = [1.0 for _ in intervals]
            self._weights_memo[intervals] = weights
        out = random.choices(intervals, weights=weights, k=n)
        if n == 1:
            return out[0]
        return out


if __name__ == "__main__":
    lambdas = [2**i for i in range(-5, 2)]
    fig, ax = plt.subplots()
    for lda in lambdas:
        ic = IntervalChooser(lambda_=lda)
        ic.plot_weights(ax=ax)
    plt.legend()
    plt.show()
