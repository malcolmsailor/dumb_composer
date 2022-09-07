from copy import copy
from fractions import Fraction
import math
from numbers import Number
import operator
import random

import typing as t

from functools import cached_property, reduce
from types import MappingProxyType

from dumb_composer.constants import METER_CONDITIONS, TIME_TYPE
from .time_utils import get_onsets_within_duration


def rhythm_method(**kwargs):
    # TODO this could be rewritten on the simpler and clearer model of
    #   pattern_method() in patterns.py
    def with_constraints(f):
        allowed_kwargs = [f"not_{condition}" for condition in METER_CONDITIONS]
        assert all(kwarg in allowed_kwargs for kwarg in kwargs)
        for kwarg in allowed_kwargs:
            setattr(f, kwarg, kwargs[kwarg] if kwarg in kwargs else False)
        setattr(f, "rhythm_method", True)
        return f

    return with_constraints


def empty_avoider(release_finder: str):
    def decorator(f):
        def wrap(
            self, onset, release, *args, avoid_empty: bool = False, **kwargs
        ):
            out = f(self, onset, release, *args, **kwargs)
            if not out and avoid_empty:
                out = [
                    {
                        "onset": onset,
                        "weight": self.meter.weight(onset),
                        "release": min(
                            release,
                            getattr(self.meter, release_finder)(
                                onset, inclusive=False
                            ),
                        ),
                    }
                ]
            return out

        return wrap

    return decorator


class TimeClass:
    _ts_dict = MappingProxyType(
        {
            "4/2": (4, 2),
            "3/2": (3, 2),
            "2/2": (2, 2),
            # Odd meters are not implemented
            # "5/4": (5, 1),
            "4/4": (4, 1),
            "3/4": (3, 1),
            "2/4": (2, 1),
            "4/8": (4, 0.5),
            "3/8": (3, 0.5),
            "2/8": (2, 0.5),
            # NB Fast 3/8 not implemented
            # "3/8": (1, 1.5),
            "6/8": (2, 1.5),
            "9/8": (3, 1.5),
            "12/8": (4, 1.5),
            "12/16": (4, 0.75),
            "6/4": (2, 3),
            "9/4": (3, 3),
            "12/4": (4, 3),
            "6/2": (2, 6),
            "9/2": (3, 6),
            "12/2": (4, 6),
        }
    )

    @classmethod
    @property
    def available_time_signatures(cls) -> t.Tuple[str]:
        return tuple(cls._ts_dict.keys())

    @staticmethod
    def _get_bounds(onset=None, release=None):
        if onset is None:
            onset = TIME_TYPE(0)
        return onset, release

    @staticmethod
    def _first_after(
        threshold: Number,
        grid_length: Number,
        inclusive: bool = True,
    ):
        """
        >>> TimeClass()._first_after(1.0, 1.0)
        1.0
        >>> TimeClass()._first_after(1.0, 1.0, inclusive=False)
        2.0
        >>> TimeClass()._first_after(1.5, 1.25)
        2.5
        """
        if inclusive:
            return math.ceil(threshold / grid_length) * grid_length
        return math.floor((threshold / grid_length) + 1) * grid_length


class MeterError(Exception):
    pass


# TODO what to do about weight of triplets?


class Meter(TimeClass):
    """
    >>> four_four = Meter("4/4")
    >>> four_four.beat_dur
    Fraction(1, 1)
    >>> four_four.semibeat_dur
    Fraction(1, 2)
    >>> four_four.superbeat_dur
    Fraction(2, 1)
    >>> four_four.bar_dur
    Fraction(4, 1)

    The weight of "beat" is always 0 so we can get the duration of smaller
    units as follows:

    >>> four_four.weight_to_grid[-2]
    Fraction(1, 4)

    Other meters, for reference. If the denominator changes the durations are
    scaled up or down by factors of two as appropriate.

    >>> three_four = Meter("3/4")
    >>> three_four.beat_dur
    Fraction(1, 1)
    >>> three_four.semibeat_dur
    Fraction(1, 2)
    >>> three_four.superbeat_dur
    Fraction(3, 1)
    >>> three_four.bar_dur
    Fraction(3, 1)

    >>> six_eight = Meter("6/8")
    >>> six_eight.beat_dur
    Fraction(3, 2)
    >>> six_eight.semibeat_dur
    Fraction(1, 2)
    >>> six_eight.superbeat_dur
    Fraction(3, 1)
    >>> six_eight.bar_dur
    Fraction(3, 1)

    >>> nine_eight = Meter("9/8")
    >>> nine_eight.beat_dur
    Fraction(3, 2)
    >>> nine_eight.semibeat_dur
    Fraction(1, 2)
    >>> nine_eight.superbeat_dur
    Fraction(9, 2)
    >>> nine_eight.bar_dur
    Fraction(9, 2)

    >>> nine_eight = Meter("12/8")
    >>> nine_eight.beat_dur
    Fraction(3, 2)
    >>> nine_eight.semibeat_dur
    Fraction(1, 2)
    >>> nine_eight.superbeat_dur
    Fraction(3, 1)
    >>> nine_eight.bar_dur
    Fraction(6, 1)
    """

    # this class should return the metric strength (as an integer, where 0 is
    # tactus) of a given time point

    def __repr__(self):
        return f"{self.__class__.__name__}('{self._ts_str}')"

    def __init__(self, ts_str, min_weight: t.Union[int, Number] = -3):
        """
        If min_weight is an int, it specifies the min_weight directly.
        >>> meter1 = Meter("4/2", min_weight=-4)

        Otherwise, min_weight is calculated as the weight of the duration
        provided (down to a minimum of -100):
        >>> meter2 = Meter("4/2", min_weight=0.125)
        >>> assert meter1.min_weight == meter2.min_weight
        """
        self._ts_str = ts_str
        try:
            n_beats, beat_dur = self._ts_dict[ts_str]
        except KeyError:
            raise MeterError(f"unsupported meter {ts_str}")
        self._n_beats = int(n_beats)
        self._beat_dur = TIME_TYPE(beat_dur)
        self._total_dur = self._n_beats * self._beat_dur
        self._compound = bool(math.log2(self._beat_dur) % 1)
        self._triple = not bool(self._n_beats / 3 % 1)
        self._semibeat_dur = (
            self._beat_dur / 3 if self._compound else self._beat_dur / 2
        )
        self._superbeat_dur = (
            self._beat_dur * 3 if self._triple else self._beat_dur * 2
        )
        self._weight_memo = {}
        self._odd_memo: t.Dict[Number, bool] = {}
        if isinstance(min_weight, int):
            self._min_weight = min_weight
        else:
            # We need to temporarily set min_weight for the self.weight
            #   algorithm to work
            self._min_weight = -100
            self._min_weight = self.weight(min_weight)

    @property
    def ts_str(self):
        return self._ts_str

    @cached_property
    def weight_to_grid(self) -> t.Dict[int, TIME_TYPE]:
        """
        >>> sorted(
        ...     (weight, float(grid))
        ...     for (weight, grid) in Meter("4/4").weight_to_grid.items()
        ... )
        [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.0), (1, 2.0), (2, 4.0)]

        >>> sorted(
        ...     (weight, float(grid))
        ...     for (weight, grid) in Meter("3/4").weight_to_grid.items()
        ... )
        [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.0), (1, 3.0)]

        >>> sorted(
        ...     (weight, float(grid))
        ...     for (weight, grid) in Meter("12/8").weight_to_grid.items()
        ... )
        [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 1.5), (1, 3.0), (2, 6.0)]
        """
        out = {}
        out[self.beat_weight] = self.beat_dur
        large_dur = self.superbeat_dur
        large_weight = self.weight(self.superbeat_dur)
        while True:
            out[large_weight] = large_dur
            if large_weight == self.max_weight:
                break
            large_dur *= 2
            large_weight = self.weight(large_dur)
        small_dur = self.semibeat_dur
        small_weight = self.weight(self.semibeat_dur)
        while True:
            out[small_weight] = small_dur
            if small_weight == self._min_weight:
                break
            small_dur /= 2
            small_weight = self.weight(small_dur)
        return out

    @property
    def scale_factor(self):
        """
        The amount by which a meter should be scaled to compare it to the
        "standard" meters with 4 in the denominator (or 8, in the case of
        compound meters).

        >>> two_two = Meter("2/2")
        >>> assert two_two.bar_dur / two_two.scale_factor == Meter("2/4").bar_dur
        """
        if self._compound:
            return 2 / 3 * self._beat_dur
        return self._beat_dur

    @cached_property
    def beat_weight(self):
        return self.weight(self._beat_dur)

    @cached_property
    def semibeat_weight(self):
        return self.weight(self._semibeat_dur)

    @cached_property
    def superbeat_weight(self):
        return self.weight(self._superbeat_dur)

    @cached_property
    def max_weight(self):
        return self.weight(0)

    @property
    def min_weight(self):
        return self._min_weight

    @property
    def beat_dur(self):  # pylint: disable=missing-docstring
        """
        >>> Meter("4/4").beat_dur
        Fraction(1, 1)
        >>> Meter("6/8").beat_dur
        Fraction(3, 2)
        """
        return self._beat_dur

    @cached_property
    def bar_dur(self):
        return self.beat_dur * self._n_beats

    @property
    def semibeat_dur(self):  # pylint: disable=missing-docstring
        """
        >>> Meter("4/4").semibeat_dur
        Fraction(1, 2)
        >>> Meter("6/8").semibeat_dur
        Fraction(1, 2)
        """
        return self._semibeat_dur

    @property
    def superbeat_dur(self):  # pylint: disable=missing-docstring
        """
        >>> Meter("4/4").superbeat_dur
        Fraction(2, 1)
        >>> Meter("6/8").superbeat_dur
        Fraction(3, 1)
        """
        return self._superbeat_dur

    @property
    def stutterbeat_durs(self) -> t.Tuple[TIME_TYPE, TIME_TYPE]:
        """
        >>> Meter("3/4").stutterbeat_durs
        (Fraction(2, 1), Fraction(1, 1))
        >>> Meter("6/8").stutterbeat_durs
        Traceback (most recent call last):
        ValueError: stutterbeat only valid for triple meters

        For now, at least, stutterbeats aren't taken into consideration
        for weight calculations. (Note that two has the same weight as
        one despite being a "stutterbeat".)
        >>> Meter("3/4").weight(1)
        0
        >>> Meter("3/4").weight(2)
        0
        """
        if not self._triple:
            raise ValueError("stutterbeat only valid for triple meters")
        return (2 * self._beat_dur, self._beat_dur)

    @property
    def stuttersemibeat_durs(self) -> t.Tuple[TIME_TYPE, TIME_TYPE]:
        """
        >>> Meter("6/8").stuttersemibeat_durs
        (Fraction(1, 1), Fraction(1, 2))
        >>> Meter("3/4").stuttersemibeat_durs
        Traceback (most recent call last):
        ValueError: stuttersemibeat only valid for compound meters
        """
        if not self._compound:
            raise ValueError("stuttersemibeat only valid for compound meters")
        return (2 * self._semibeat_dur, self._semibeat_dur)

    @property
    def is_compound(self):
        """
        is_compound is True when beat_dur / sub_beat_dur == 3

        >>> Meter("4/4").is_compound
        False
        >>> Meter("6/8").is_compound
        True
        """
        return self._compound

    @property
    def is_triple(self):
        """
        is_triple is True when bar_dur / beat_dur == 2

        >>> Meter("9/8").is_triple
        True
        >>> Meter("6/8").is_triple
        False
        """
        return self._triple

    @property
    def is_duple(self):
        """
        >>> Meter("3/4").is_duple
        False
        >>> Meter("6/8").is_duple
        True
        """
        return not self._triple

    @cached_property
    def non_odd_lengths(self):
        """See the method 'duration_is_odd'."""
        out = set()
        demisemibeat = self.weight_to_grid[-2]
        out.update({demisemibeat, demisemibeat * 3})
        if self._compound:
            out.update({self.semibeat_dur})
            out.update(
                {
                    self.stuttersemibeat_durs[0],
                    self.beat_dur + self.stuttersemibeat_durs[0],
                }
            )
        else:
            out.update({self.semibeat_dur, self.semibeat_dur * 3})
        if self._triple:
            out.update({self.beat_dur})
            out.update(
                {
                    self.stutterbeat_durs[0],
                    self.bar_dur + self.stutterbeat_durs[0],
                }
            )
        else:
            out.update({self.beat_dur, self.beat_dur * 3})
            if self._n_beats > 2:
                out.update({self.superbeat_dur, 3 * self.superbeat_dur})
        out.update({self.bar_dur, 2 * self.bar_dur})
        return out

    def __call__(self, time):
        """
        >>> four_four = Meter("4/4")
        >>> four_four(0.0)
        2
        >>> four_four(2.0)
        1
        >>> four_four(1.0)
        0
        >>> four_four(0.5)
        -1

        >>> nine_eight = Meter("9/8")
        >>> nine_eight(0.0)
        1
        >>> nine_eight(1.5)
        0
        >>> nine_eight(3.0)
        0
        >>> nine_eight(0.5)
        -1
        >>> nine_eight(1.0)
        -1
        """
        return self.weight(time)

    @property
    def n_beats(self):  # pylint: disable=missing-docstring
        return self._n_beats

    def _duple_weight(self, time, n_beats=None):
        if n_beats is None:
            n_beats = self._n_beats
        time /= self._beat_dur
        exp = math.ceil(math.log2(n_beats))
        while exp > self._min_weight:
            if not time % 2**exp:
                break
            exp -= 1
        return exp

    def _triple_weight(self, time):
        if time == 0:
            return 1
        return self._duple_weight(time, n_beats=1)

    def _compound_weight(self, time):
        if time % self._beat_dur == 0:
            if self._triple:
                return self._triple_weight(time)
            else:
                return self._duple_weight(time)
        normalize_factor = self._beat_dur / 1.5
        time /= normalize_factor
        exp = -1
        while exp > self._min_weight:
            if not time % 2**exp:
                break
            exp -= 1
        return exp

    def weight(self, time):
        time = TIME_TYPE(time)
        time %= self._total_dur
        if time in self._weight_memo:
            return self._weight_memo[time]

        if self._compound:
            out = self._compound_weight(time)
        elif self._triple:
            out = self._triple_weight(time)
        else:
            out = self._duple_weight(time)
        self._weight_memo[time] = out
        return out

    def weights_between(
        self,
        grid_length: Number,
        onset: Number,
        release: Number,
        include_start: bool = True,
        include_end: bool = False,
        out_format: str = "list",
        include_releases: bool = False,
    ) -> t.Union[t.Dict[Number, Number], t.List[t.Dict[str, Number]]]:
        """
        >>> two_four = Meter("2/4")
        >>> two_four.weights_between(0.5, 0.5, 1.5)
        [{'onset': 0.5, 'weight': -1}, {'onset': 1.0, 'weight': 0}]

        >>> twelve_eight = Meter("12/8")
        >>> twelve_eight.weights_between(0.5, 3.0, 6.1, out_format="dict")
        {3.0: 1, 3.5: -1, 4.0: -1, 4.5: 0, 5.0: -1, 5.5: -1, 6.0: 2}
        """
        if include_releases and out_format == "dict":
            raise ValueError(
                "`include_releases` only valid with `out_format` == 'list'"
            )
        onset, release = self._get_bounds(onset, release)
        time = self._first_after(onset, grid_length, inclusive=include_start)
        out = {} if out_format == "dict" else []
        op = operator.le if include_end else operator.lt
        while op(time, release):
            if out_format == "dict":
                out[time] = self(time)
            elif include_releases:
                out.append(
                    {
                        "onset": time,
                        "weight": self(time),
                        "release": min(time + grid_length, release),
                    }
                )
            else:
                out.append({"onset": time, "weight": self(time)})
            time += grid_length
        return out

    def onsets_between(
        self,
        start: Number,
        stop: Number,
        min_weight: int,
        include_start: bool = True,
        include_stop: bool = False,
        out_format: str = "list",
    ) -> t.Union[t.Dict[Number, Number], t.List[t.Dict[str, Number]]]:
        """
        >>> two_four = Meter("2/4")
        >>> two_four.onsets_between(0, 1, -2, out_format="dict")
        {Fraction(0, 1): 1, Fraction(1, 4): -2, Fraction(1, 2): -1, Fraction(3, 4): -2}
        """
        grid = self.weight_to_grid[min_weight]
        return self.weights_between(
            grid,
            start,
            stop,
            include_start,
            include_stop,
            out_format=out_format,
        )

    def beats_between(self, *args, **kwargs):
        return self.weights_between(self._beat_dur, *args, **kwargs)

    def semibeats_between(self, *args, **kwargs):
        return self.weights_between(self._semibeat_dur, *args, **kwargs)

    def superbeats_between(self, *args, **kwargs):
        return self.weights_between(self._superbeat_dur, *args, **kwargs)

    def beat_after(self, time: Number, inclusive: bool = True):
        return self._first_after(time, self.beat_dur, inclusive=inclusive)

    def semibeat_after(self, time: Number, inclusive: bool = True):
        return self._first_after(time, self.semibeat_dur, inclusive=inclusive)

    def superbeat_after(self, time: Number, inclusive: bool = True):
        return self._first_after(time, self.superbeat_dur, inclusive=inclusive)

    def get_onset_of_greatest_weight_between(
        self,
        start: Number,
        stop: Number,
        include_start: bool = True,
        include_stop: bool = False,
        return_first: bool = False,
    ) -> t.Tuple[TIME_TYPE, int]:
        """
        >>> nine_eight = Meter("9/8")
        >>> nine_eight.get_onset_of_greatest_weight_between(4.5, 9.0)
        (Fraction(9, 2), 1)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     4.5, 9.0, include_start=False)
        (Fraction(15, 2), 0)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     4.5, 9.0, include_stop=True)
        (Fraction(9, 1), 1)

        If `return_first` is True, then in the event of a tie (which can occur
        between downbeats, or between the 2nd and 3rd pulses in triple
        divisions), we take the first item. Otherwise, if there are two items,
        we take the second one.

        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.5, 1.5)
        (Fraction(1, 1), -1)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.5, 1.5, return_first=True)
        (Fraction(1, 2), -1)

        If the interval is several measures or more long, there may be a tie
        between many downbeats. In this case, if `return_first` is False, we
        take middle downbeat (if there are an odd number), or the middle + 1th
        downbeat (if there are an even number).

        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.0, 13.5) # first 3 measures, returns downbeat of measure 2
        (Fraction(9, 2), 1)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.0, 18.0) # first 4 measures, returns downbeat of measure 3
        (Fraction(9, 1), 1)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.0, 18.0, return_first=True) # returns downbeat of measure 1
        (Fraction(0, 1), 1)
        """
        for weight in range(self.max_weight, self._min_weight - 1, -1):
            grid = self.weight_to_grid[weight]
            onsets = self.weights_between(
                grid,
                start,
                stop,
                include_start,
                include_stop,
                out_format="list",
            )
            if onsets:
                break
        if not onsets:
            raise MeterError(
                f"no onsets between {start} and {stop} with weight greater "
                f"than self.min_weight={self.min_weight}"
            )
        if len(onsets) == 1 or return_first:
            return tuple(onsets[0].values())
        if len(onsets) > 2:
            assert all(onset["weight"] == self.max_weight for onset in onsets)
            return tuple(onsets[math.floor(len(onsets) / 2)].values())
        return tuple(onsets[1].values())

    def split_at_metric_strong_points(
        self,
        items: t.List[t.Any],
        min_split_dur: t.Optional[Number] = None,
        force_split: bool = False,
    ) -> t.List[t.Any]:
        """
        Items in input list must have mutable "onset" and "release" attributes
        and they must be comparable for equality.

        The items will be shallow-copied.

        >>> class Dur:
        ...     def __init__(self, onset, release):
        ...         self.onset, self.release = onset, release
        ...
        ...     def __repr__(self):
        ...         return f"{float(self.onset)}_to_{float(self.release)}"
        ...
        ...     def __eq__(self, other):
        ...         return self.onset == other.onset and self.release == other.release

        >>> four_four = Meter("4/4")
        >>> four_four.split_at_metric_strong_points(
        ...     [Dur(0, 9), Dur(9, 14), Dur(14, 18)])
        [0.0_to_4.0, 4.0_to_8.0, 8.0_to_9.0, 9.0_to_10.0, 10.0_to_12.0, 12.0_to_14.0, 14.0_to_16.0, 16.0_to_18.0]

        The leading portion will be split recursively down to `min_split_dur`
        (if passed).

        >>> four_four.split_at_metric_strong_points([Dur(0.25, 2)])
        [0.25_to_0.5, 0.5_to_1.0, 1.0_to_2.0]
        >>> four_four.split_at_metric_strong_points([Dur(0.25, 2)],
        ...     min_split_dur=1.0)
        [0.25_to_1.0, 1.0_to_2.0]
        >>> four_four.split_at_metric_strong_points([Dur(1, 4)],
        ...     min_split_dur=1.0)
        [1.0_to_2.0, 2.0_to_4.0]

        Note that "odd" durations like the following are not avoided. For that,
        see TODO

        >>> four_four.split_at_metric_strong_points([Dur(0, 3.5)])
        [0.0_to_3.5]
        >>> four_four.split_at_metric_strong_points([Dur(0.0, 1.75)])
        [0.0_to_1.75]

        If `force_split` is True, then at least one split will be made.
        >>> four_four.split_at_metric_strong_points(
        ...     [Dur(0, 3.5)], force_split=True)
        [0.0_to_2.0, 2.0_to_3.5]
        """

        def _inner_sub(
            item: t.Any, at_least_one_split: bool = False
        ) -> t.List[t.Any]:
            out = []
            # for item in items:
            start_onset = item.onset
            if (
                min_split_dur is not None
                and item.release - item.onset <= min_split_dur
            ):
                out.append(copy(item))
                return out
            start_weight = self.weight(item.onset)
            try:
                (
                    mid_onset,
                    mid_weight,
                ) = self.get_onset_of_greatest_weight_between(
                    item.onset,
                    item.release,
                    include_start=False,
                    return_first=True,
                )
            except MeterError:
                # there are no weights between start_onset and item.release
                pass
            else:
                while mid_weight >= start_weight or at_least_one_split:
                    new_item = copy(item)
                    new_item.onset = start_onset
                    new_item.release = mid_onset
                    out.append(new_item)
                    start_onset, start_weight = mid_onset, mid_weight
                    if (
                        min_split_dur is not None
                        and item.release - start_onset <= min_split_dur
                    ):
                        break
                    try:
                        (
                            mid_onset,
                            mid_weight,
                        ) = self.get_onset_of_greatest_weight_between(
                            start_onset,
                            item.release,
                            include_start=False,
                            return_first=True,
                        )
                    except MeterError:
                        # there are no weights between start_onset and
                        # item.release
                        break
                    at_least_one_split = False
            final_item = copy(item)
            final_item.onset = start_onset
            out.append(final_item)
            return out

        result = []
        for item in items:
            out = _inner_sub(item, at_least_one_split=force_split)
            if not out:
                continue
            initial, remainder = out[0], out[1:]
            while (split_initial := _inner_sub(initial)) != [initial]:
                out = split_initial + remainder
                initial, remainder = out[0], out[1:]
            result.extend(out)
        return result

    def split_odd_duration(
        self, item: t.Any, min_split_dur: t.Optional[Number] = None
    ) -> t.List[t.Any]:
        """
        Items in input list must have mutable "onset" and "release" attributes
        and they must be comparable for equality.

        The items will be shallow-copied.

        >>> class Dur:
        ...     def __init__(self, onset, release):
        ...         self.onset, self.release = onset, release
        ...
        ...     def __repr__(self):
        ...         return f"{float(self.onset)}_to_{float(self.release)}"
        ...
        ...     def __eq__(self, other):
        ...         return self.onset == other.onset and self.release == other.release

        >>> four_four = Meter("4/4")
        >>> four_four.split_odd_duration(Dur(0, 1.25))
        [0.0_to_1.0, 1.0_to_1.25]
        """
        try:
            if not self.duration_is_odd(item.release - item.onset):
                return [item]
        except ValueError:
            # duration has weight < -3
            pass
        return self.split_at_metric_strong_points(
            [item], min_split_dur=min_split_dur, force_split=True
        )

    def duration_is_odd(self, duration: Number) -> bool:
        """

        "Odd" durations are difficult to describe. In purely duple (i.e.,
        non-compound) meters, an 'odd' duration is one that is not either
            - 2**n times a metric division (where n is an integer)
            - 3 * 2**n times a metric division (where n is an integer)
        This definition doesn't work for other meters though.

        >>> four_four = Meter("4/4")
        >>> four_four.duration_is_odd(2)
        False
        >>> any(
        ...     four_four.duration_is_odd(dur)
        ...     for dur in (1/4, 1/2, 3/4, 1, 3/2, 2, 3, 4, 6, 8, 32, 96, 512)
        ... )
        False
        >>> four_four.duration_is_odd(5)
        True
        >>> all(
        ...     four_four.duration_is_odd(dur)
        ...     for dur in (5/4, 5/2, 5, 10, 7/4, 7/2, 7, 9/4, 9/2, 9, 13/4, 17/4)
        ... )
        True

        >>> nine_four = Meter("9/4")
        >>> nine_four.duration_is_odd(9)
        False

        Will raise a ValueError if called on a duration at less than the
        'demisemibeat' level (i.e., metric weight -3 or smaller).

        """
        if duration in self.non_odd_lengths:
            return False
        if self.weight(duration) < -2:
            raise ValueError(
                f"Minimum metric weight for Meter.duration_is_odd is -2; "
                f"got {self.weight(duration)}"
            )
        return bool(duration % self.bar_dur)


class RhythmFetcher(TimeClass):
    # What this class should ideally do is take a time signature (best way of
    # representing this?) and then return various "rhythms" (e.g., offbeat 8ths;
    # trochees; dactyls; etc.) adapted to that time signature

    def __init__(self, ts: t.Union[Meter, str]):
        if isinstance(ts, str):
            self.meter = Meter(ts)
        else:
            self.meter = ts
        self._rhythms = self._filter_rhythms()

    def __call__(self, rhythm_type=None, onset=None, release=None):
        onset, release = self._get_bounds(onset, release)
        if rhythm_type is None:
            rhythm_type = random.choice(self._rhythms)
        return getattr(self, rhythm_type)(onset, release)

    def __getattr__(self, name):
        try:
            return getattr(self.meter, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def _filter_rhythms(self):
        out = []
        for name in self._all_rhythms:
            f = getattr(self, name)
            if any(
                (
                    getattr(self.meter, f"is_{condition}")
                    and getattr(f, f"not_{condition}")
                )
                for condition in METER_CONDITIONS
            ):
                continue
            out.append(name)
        return out

    def _regularly_spaced(self, length, onset, release):
        return self.meter.weights_between(
            length, onset, release, include_releases=True
        )
        # time = self._first_after(onset, length)
        # out = []
        # while time < release:
        #     out.append(
        #         {
        #             "onset": time,
        #             "release": min(time + length, release),
        #             "weight": self.meter(time),
        #         }
        #     )
        #     time += length
        # return out

    def _regularly_spaced_by_weight(self, weight, onset, release):
        grid_length = self.meter.weight_to_grid[weight]
        return self.meter.weights_between(
            grid_length, onset, release, include_releases=True
        )

    @rhythm_method()
    @empty_avoider("beat_after")
    def beats(self, onset, release):
        return self._regularly_spaced(self.beat_dur, onset, release)

    @rhythm_method()
    @empty_avoider("semibeat_after")
    def semibeats(self, onset, release):
        return self._regularly_spaced(self.semibeat_dur, onset, release)

    @rhythm_method()
    @empty_avoider("superbeat_after")
    def superbeats(self, onset, release):
        return self._regularly_spaced(self.superbeat_dur, onset, release)

    @rhythm_method(not_triple=True, not_compound=True)
    @empty_avoider("semibeat_after")
    def amphibrach(self, onset, release):
        semibeats = self.meter.semibeats_between(onset, release)
        on_weight = self.meter.beat_weight + 1
        off_weight = self.meter.beat_weight - 1
        return [
            beat
            | {"release": min(beat["onset"] + self.meter.semibeat_dur, release)}
            for beat in semibeats
            if beat["weight"] >= on_weight or beat["weight"] <= off_weight
        ]

    @rhythm_method(not_triple=True)
    @empty_avoider("semibeat_after")
    def trochee(self, onset, release):
        semibeats = self.meter.semibeats_between(onset, release)
        if self.is_compound:
            on_weight = self.meter.beat_weight
        else:
            on_weight = self.meter.beat_weight + 1
        out = []
        append_next = False
        for sb in semibeats:
            if append_next:
                out.append(
                    sb
                    | {
                        "release": min(
                            release, sb["onset"] + self.meter.semibeat_dur
                        )
                    }
                )
                append_next = False
            elif sb["weight"] >= on_weight:
                append_next = True
                out.append(
                    sb
                    | {
                        "release": min(
                            release, sb["onset"] + self.meter.semibeat_dur
                        )
                    }
                )
        return out

    @rhythm_method(not_triple=True)
    @empty_avoider("semibeat_after")
    def iamb(self, onset, release):
        inclusive = release % self.meter.beat_dur == 0
        semibeats = self.meter.semibeats_between(
            onset, release, include_end=inclusive
        )
        if self.is_compound:
            on_weight = self.meter.beat_weight
        else:
            on_weight = self.meter.beat_weight + 1
        out = []
        append_next = False
        for i, sb in enumerate(reversed(semibeats)):
            if append_next:
                out.append(
                    sb
                    | {
                        "release": min(
                            release, sb["onset"] + self.meter.semibeat_dur
                        )
                    }
                )
                append_next = False
            elif sb["weight"] >= on_weight:
                append_next = True
                if not inclusive or not i == 0:
                    out.append(
                        sb
                        | {
                            "release": min(
                                release, sb["onset"] + self.meter.semibeat_dur
                            )
                        }
                    )
        out.reverse()
        return out

    @rhythm_method()
    @empty_avoider("semibeat_after")
    def off_semibeats(self, onset, release):
        semibeats = self.meter.semibeats_between(onset, release)
        omit_weight = self.meter.beat_weight + 1
        out = [
            sb
            | {"release": min(release, sb["onset"] + self.meter.semibeat_dur)}
            for sb in semibeats
            if sb["weight"] < omit_weight
        ]
        return out

    @rhythm_method()
    @empty_avoider("beat_after")
    def off_beats(self, onset, release):
        beats = self.meter.beats_between(onset, release)
        omit_weight = self.meter.max_weight
        out = [
            sb | {"release": min(release, sb["onset"] + self.meter.beat_dur)}
            for sb in beats
            if sb["weight"] < omit_weight
        ]
        return out

    def next_n(
        self,
        weight_or_name: t.Union[int, str],
        onset: Number,
        n: int,
        release: t.Optional[Number] = None,
    ):
        """
        >>> rf = RhythmFetcher("4/4")
        >>> rf.next_n("semibeat", onset=3.25, n=3)
        [Fraction(7, 2), Fraction(4, 1), Fraction(9, 2)]
        >>> rf.next_n("semibeat", onset=3.25, n=3, release=4.0)
        [Fraction(7, 2)]
        """
        if isinstance(weight_or_name, str):
            weight = getattr(self.meter, f"{weight_or_name}_weight")
        else:
            weight = weight_or_name
        out = []
        grid = self.meter.weight_to_grid[weight]
        for i in range(n):
            next_onset = self._first_after(onset, grid, inclusive=i == 0)
            if release is not None and next_onset >= release:
                break
            out.append(next_onset)
            onset = next_onset
        return out

    # The definition of _all_rhythms should be the last line in the class
    # definition
    _all_rhythms = tuple(
        name
        for name, f in locals().items()
        if getattr(f, "rhythm_method", False)
    )


# doctests in cached_property methods are not discovered and need to be
#   added explicitly to __test__; see https://stackoverflow.com/a/72500890/10155119
__test__ = {"Meter.weight_to_grid": Meter.weight_to_grid}
