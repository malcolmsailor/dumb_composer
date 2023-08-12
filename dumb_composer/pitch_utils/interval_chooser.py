"""Provides a class, DeprecatedIntervalChooser, that chooses small intervals more often than 
larger intervals."""
import logging
import math
import random
import typing as t
from abc import abstractmethod
from collections import Counter, defaultdict  # used by doctests
from dataclasses import dataclass, field
from statistics import fmean, harmonic_mean

import matplotlib.pyplot as plt
import numpy as np

from dumb_composer.constants import TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS
from dumb_composer.pitch_utils.types import ChromaticInterval, SettingsBase, Weight
from dumb_composer.utils.math_ import softmax
from dumb_composer.utils.shell_plot import print_bar  # used by doctests

NonnegativeFloat = float

# TODO: (Malcolm 2023-07-21) perhaps combine the approach of giving a score proportional
#   to size *and* an arbitrary score

LOGGER = logging.getLogger(__name__)


@dataclass
class _IntervalChooserBaseSettings(SettingsBase):
    weight_harmonic_intervals: bool = False
    harmonic_interval_weights: dict[ChromaticInterval, Weight] | None = field(
        default_factory=lambda: TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS.copy()
    )

    def __post_init__(self):
        if not self.weight_harmonic_intervals:
            self.harmonic_interval_weights = None


class _IntervalChooserBase:
    def __init__(self, settings: _IntervalChooserBaseSettings):
        self._weights_memo = {}
        self._weight_harmonic_intervals = settings.weight_harmonic_intervals
        harmonic_interval_weights = settings.harmonic_interval_weights
        if harmonic_interval_weights is None:
            self._harmonic_interval_weights = None
        else:
            softmaxed_weights = softmax(list(harmonic_interval_weights.values()))
            self._harmonic_interval_weights = {
                k: v
                for k, v in zip(harmonic_interval_weights.keys(), softmaxed_weights)
            }

    @abstractmethod
    def _get_melodic_interval_weight(self, interval) -> float:
        raise NotImplementedError

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
        """
        We take the product of each weight in weights. This seems to be better than the
        sum because with the sum, if one weight is very low but the other is high, the
        result is high, and so we can end up choosing items that should be disfavored.
        """
        filtered_weights = [w for w in weights if w is not None]
        if len(filtered_weights) > 1:
            # consolidated_weights = [math.prod(ws) for ws in zip(*filtered_weights)]
            consolidated_weights = [fmean(ws) for ws in zip(*filtered_weights)]

        else:
            consolidated_weights = filtered_weights[0]

        return random.choices(intervals, weights=consolidated_weights, k=n)

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
        elif self._harmonic_interval_weights is not None:
            harmonic_interval_weights = [
                self._harmonic_interval_weights[h % 12] for h in harmonic_intervals
            ]
        else:
            if self._weight_harmonic_intervals:
                LOGGER.warning(
                    f"{harmonic_intervals=} but {self._harmonic_interval_weights=}"
                )
            harmonic_interval_weights = None

        return self._make_weighted_choice(
            list(range(len(melodic_intervals))),
            melodic_interval_weights,
            harmonic_interval_weights,
            custom_weights,
            n=n,
        )

    def choose_intervals(
        self,
        intervals: t.Sequence[int],
        *,
        harmonic_intervals: t.Sequence[int] | None = None,
        n: int = 1,
        custom_weights: t.Sequence[float] | None = None,
    ) -> t.List[int]:
        """
        For now we are just summing `custom_weights` with `interval_weights`.
        The latter are normalized so they always sum to 1.
        """
        indices = self.get_interval_indices(
            intervals, harmonic_intervals, n, custom_weights
        )
        return [intervals[i] for i in indices]
        # intervals = tuple(intervals)

        # interval_weights = self._get_melodic_interval_weights(intervals)

        # return self._make_weighted_choice(
        #     intervals, interval_weights, custom_weights, n=n
        # )

    def __call__(
        self,
        intervals: t.Sequence[int],
        *,
        harmonic_intervals: t.Sequence[int] | None = None,
        custom_weights: t.Sequence[float] | None = None,
    ) -> int:
        return self.choose_intervals(
            intervals,
            harmonic_intervals=harmonic_intervals,
            n=1,
            custom_weights=custom_weights,
        )[0]


@dataclass
class DeprecatedIntervalChooserSettings(_IntervalChooserBaseSettings):
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


class DeprecatedIntervalChooser(_IntervalChooserBase):
    """
    This class uses a function (similar to an exponential
    PDF) to score melodic intervals. However it seems better to me to simply provide
    scores for each interval, as in the revised IntervalChooser, since the
    marginal probability of a melodic interval isn't merely a function of its size
    (compare the probability of an octave to the probability of a major seventh).

    ------------------------------------------------------------------------------------
    Default settings
    ------------------------------------------------------------------------------------

    >>> ic = DeprecatedIntervalChooser()
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
    >>> ic = DeprecatedIntervalChooser(
    ...     DeprecatedIntervalChooserSettings(smaller_mel_interval_concentration=1.0)
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({0: 775, 2: 109, -2: 103, 5: 6, -5: 4, 7: 2, -7: 1})

    Zero smaller_mel_interval_concentration (uniform distribution):
    >>> ic = DeprecatedIntervalChooser(
    ...     DeprecatedIntervalChooserSettings(smaller_mel_interval_concentration=0.0)
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-7: 156, 7: 153, -5: 150, 0: 146, 5: 135, 2: 132, -2: 128})

    Reweighting unison:
    >>> ic = DeprecatedIntervalChooser(
    ...     DeprecatedIntervalChooserSettings(
    ...         smaller_mel_interval_concentration=1.0, unison_weighted_as=7
    ...     )
    ... )
    >>> Counter(ic.choose_intervals(intervals, n=1000))  # doctest: +SKIP
    Counter({-2: 473, 2: 468, 5: 33, -5: 20, 7: 2, -7: 2, 0: 2})
    """

    def __init__(self, settings: t.Optional[DeprecatedIntervalChooserSettings] = None):
        if settings is None:
            settings = DeprecatedIntervalChooserSettings()
        super().__init__(settings)
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

    def _get_melodic_interval_weight(self, interval) -> float:
        # Exponential distribution is lambda * exp(-lambda * x)
        # However, scaling by lambda is just done to normalize the distribution
        # so it integrates to 1. For a number of reasons, this isn't necessary here.
        return math.exp(
            -self._smaller_mel_interval_concentration
            * (self._unison_weighted_as if interval == 0 else abs(interval))
        )


@dataclass
class IntervalChooserSettings(_IntervalChooserBaseSettings):
    """
    By default, negative intervals receive the same weight as positive intervals. (This
    occurs in the post-init function.)
    >>> IntervalChooserSettings(interval_weights={3: 0.75, 4: 1.0}).interval_weights
    {3: 0.75, 4: 1.0, -3: 0.75, -4: 1.0}

    However, we can override this behavior by explicitly providing negative intervals:
    >>> IntervalChooserSettings(
    ...     interval_weights={3: 0.75, 4: 1.0, -3: 1.5}
    ... ).interval_weights
    {3: 0.75, 4: 1.0, -3: 1.5, -4: 1.0}
    """

    interval_weights: dict[ChromaticInterval, float] = field(
        default_factory=lambda: {
            0: 4.0,
            1: 6.0,
            2: 6.0,
            3: 4.5,
            4: 4.0,
            5: 2.0,
            6: 0.5,
            7: 1.5,
            8: 0.5,
            -8: 0.25,
            9: 0.25,
            -9: 0.1,
            10: 0.1,
            -10: 0.01,
            11: 0.01,
            -11: 0.005,
            12: 0.5,
            -12: 0.25,
        }
    )

    # default_weight is applied to any interval not in interval_weights
    default_interval_weight: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        for interval, weight in list(self.interval_weights.items()):
            if interval > 0 and -1 * interval not in self.interval_weights:
                self.interval_weights[-1 * interval] = weight


class IntervalChooser(_IntervalChooserBase):
    """
    >>> ic = IntervalChooser()
    >>> intervals = list(range(-12, 13))

    To get a single interval, call an instance directly
    >>> ic(intervals)  # doctest: +SKIP
    1

    To get a list of intervals use `choose_intervals()`
    >>> intervals = ic.choose_intervals(intervals, n=10000)
    >>> intervals  # doctest: +SKIP
    [0, -1, -1, -6, -1, 3, 1, -5, 1, -3, -1, ...
    >>> counts = Counter(intervals)
    >>> print_bar(
    ...     "Intervals",
    ...     counts,
    ...     horizontal=True,
    ...     char_height=65,
    ...     sort_by_key=True,
    ...     file=None,
    ... )  # doctest: +SKIP
    -12 ██▋
    -11 ▏
    -10 ▏
     -9 █
     -8 ██▌
     -7 ███████████████▎
     -6 ████▌
     -5 ████████████████████▋
     -4 ███████████████████████████████████████▎
     -3 █████████████████████████████████████████████████▊
     -2 ██████████████████████████████████████████████████████████████▉
     -1 ██████████████████████████████████████████████████████████▊
      0 ███████████████████████████████████████▍
      1 █████████████████████████████████████████████████████████████████
      2 ██████████████████████████████████████████████████████████▉
      3 ██████████████████████████████████████████████▎
      4 ██████████████████████████████████████▎
      5 ████████████████████▎
      6 █████▏
      7 ██████████████▏
      8 █████▏
      9 ██▍
     10 █▏
     11 ▏
     12 █████

    """

    def __init__(self, settings: IntervalChooserSettings | None = None):
        if settings is None:
            settings = IntervalChooserSettings()
        super().__init__(settings)
        self._interval_weights = defaultdict(
            lambda: settings.default_interval_weight, settings.interval_weights
        )

    def _get_melodic_interval_weight(self, interval) -> float:
        return self._interval_weights[interval]
