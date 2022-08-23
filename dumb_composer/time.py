import math
from numbers import Number
import operator
import random

import typing as t

from functools import cached_property
from types import MappingProxyType

from dumb_composer.constants import METER_CONDITIONS, TIME_TYPE
from .time_utils import get_onsets_within_duration


def rhythm_method(**kwargs):
    def with_constraints(f):
        allowed_kwargs = [f"not_{condition}" for condition in METER_CONDITIONS]
        assert all(kwarg in allowed_kwargs for kwarg in kwargs)
        for kwarg in allowed_kwargs:
            setattr(f, kwarg, kwargs[kwarg] if kwarg in kwargs else False)
        setattr(f, "rhythm_method", True)
        return f

    return with_constraints


class TimeClass:
    _ts_dict = MappingProxyType(
        {
            "4/2": (4, 2),
            "3/2": (3, 2),
            "2/2": (2, 2),
            "5/4": (5, 1),
            "4/4": (4, 1),
            "3/4": (3, 1),
            "2/4": (2, 1),
            "4/8": (4, 0.5),
            "3/8": (3, 0.5),
            "2/8": (2, 0.5),
            "3/8": (1, 1.5),
            "6/8": (2, 1.5),
            "9/8": (3, 1.5),
            "12/8": (4, 1.5),
            "6/4": (2, 3),
            "9/4": (3, 3),
            "12/4": (4, 3),
            "6/2": (2, 6),
            "9/2": (3, 6),
            "12/2": (4, 6),
        }
    )

    @staticmethod
    def _get_bounds(onset=None, release=None, dur=None):
        if onset is None:
            onset = TIME_TYPE(0)
        if release is None:
            release = onset + dur
        return onset, release

    @staticmethod
    def _first_after(
        threshold: Number, grid_length: Number, inclusive: bool = True
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


class Meter(TimeClass):
    # this class should return the metric strength (as an integer, where 0 is
    # tactus) of a given time point

    def __repr__(self):
        return f"{self.__class__.__name__}('{self._ts_str}')"

    def __init__(self, ts_str, min_weight=-3):
        self._ts_str = ts_str
        n_beats, beat_dur = self._ts_dict[ts_str]
        self._n_beats = TIME_TYPE(n_beats)
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
        self._memo = {}
        self._min_weight = min_weight

    # TODO doctest doesn't run with cached_property (at least in pytest)
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

        # TODO fix 12/8; shouldn't it end (1, 3.0), (2, 6.0)?
        >>> sorted(
        ...     (weight, float(grid))
        ...     for (weight, grid) in Meter("12/8").weight_to_grid.items()
        ... )
        [(-3, 0.125), (-2, 0.25), (-1, 0.5), (0, 3.0), (1, 6.0)]
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

    @cached_property
    def beat_weight(self):
        return self.weight(self._beat_dur)

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
    def is_compound(self):
        """
        >>> Meter("4/4").is_compound
        False
        >>> Meter("6/8").is_compound
        True
        """
        return self._compound

    @property
    def is_triple(self):
        """
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
        if time == 0:
            return 1
        if time % self._beat_dur == 0:
            return 0
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
        if time in self._memo:
            return self._memo[time]

        if self._compound:
            out = self._compound_weight(time)
        elif self._triple:
            out = self._triple_weight(time)
        else:
            out = self._duple_weight(time)
        self._memo[time] = out
        return out

    def weights_between(
        self,
        grid_length: Number,
        onset: Number,
        release: Number,
        include_start: bool = True,
        include_end: bool = False,
        out_format: str = "list",
    ) -> t.Union[t.Dict[Number, Number], t.List[t.Dict[str, Number]]]:
        """
        >>> two_four = Meter("2/4")
        >>> two_four.weights_between(0.5, 0.5, 1.5)
        [{'onset': 0.5, 'weight': -1}, {'onset': 1.0, 'weight': 0}]

        # TODO it seems to me that 12/8 weights are wrong
        >>> twelve_eight = Meter("12/8")
        >>> twelve_eight.weights_between(0.5, 3.0, 6.1, out_format="dict")
        {3.0: 0, 3.5: -1, 4.0: -1, 4.5: 0, 5.0: -1, 5.5: -1, 6.0: 1}
        """
        onset, release = self._get_bounds(onset, release)
        time = self._first_after(onset, grid_length, inclusive=include_start)
        out = {} if out_format == "dict" else []
        op = operator.le if include_end else operator.lt
        while op(time, release):
            if out_format == "dict":
                out[time] = self(time)
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

        If the interval is several measures or more long, there may be a tie
        between many downbeats. In this case, we take middle downbeat (if there
        are an odd number), or the middle + 1th downbeat (if there are an even
        number).

        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.0, 13.5) # first 3 measures, returns downbeat of measure 2
        (Fraction(9, 2), 1)
        >>> nine_eight.get_onset_of_greatest_weight_between(
        ...     0.0, 18.0) # first 4 measures, returns downbeat of measure 3
        (Fraction(9, 1), 1)
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
        if len(onsets) > 2:
            assert all(onset["weight"] == self.max_weight for onset in onsets)
            return tuple(onsets[math.floor(len(onsets) / 2)].values())
        if len(onsets) == 1 or return_first:
            return tuple(onsets[0].values())
        return tuple(onsets[1].values())


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

    def __call__(self, rhythm_type=None, onset=None, release=None, dur=None):
        onset, release = self._get_bounds(onset, release, dur)
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
        # we assume that the regularly spaced onsets begin from zero;
        # perhaps this assumption needs to be modified?
        time = self._first_after(onset, length)
        out = []
        while time < release:
            out.append(
                {
                    "onset": time,
                    "release": min(time + length, release),
                    "weight": self.meter(time),
                }
            )
            time += length
        return out

    @rhythm_method()
    def beats(self, onset, release):
        return self._regularly_spaced(self.beat_dur, onset, release)

    @rhythm_method()
    def semibeats(self, onset, release):
        return self._regularly_spaced(self.semibeat_dur, onset, release)

    @rhythm_method()
    def superbeats(self, onset, release):
        return self._regularly_spaced(self.superbeat_dur, onset, release)

    @rhythm_method(not_triple=True, not_compound=True)
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
    def off_semibeats(self, onset, release):
        semibeats = self.meter.semibeats_between(onset, release)
        omit_weight = self.meter.beat_weight + 1
        return [
            sb
            | {"release": min(release, sb["onset"] + self.meter.semibeat_dur)}
            for sb in semibeats
            if sb["weight"] < omit_weight
        ]

    @rhythm_method()
    def off_beats(self, onset, release):
        beats = self.meter.beats_between(onset, release)
        omit_weight = self.meter.max_weight
        return [
            sb | {"release": min(release, sb["onset"] + self.meter.beat_dur)}
            for sb in beats
            if sb["weight"] < omit_weight
        ]

    @rhythm_method()
    def beats(self, onset, release):
        beats = self.meter.beats_between(onset, release)
        return [
            sb | {"release": min(release, sb["onset"] + self.meter.beat_dur)}
            for sb in beats
        ]

    # The definition of _all_rhythms should be the last line in the class
    # definition
    _all_rhythms = tuple(
        name
        for name, f in locals().items()
        if getattr(f, "rhythm_method", False)
    )
