import math
import operator
import random

from functools import cached_property
from types import MappingProxyType

from dumb_composer.constants import METER_CONDITIONS, TIME_TYPE


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
    def _first_after(threshold, length):
        return math.ceil(threshold / length) * length


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
        self._min_weight = -3

    @cached_property
    def beat_weight(self):
        return self._weight(self._beat_dur)

    @cached_property
    def max_weight(self):
        return self._weight(0)

    @property
    def beat_dur(self):  # pylint: disable=missing-docstring
        return self._beat_dur

    @property
    def semibeat_dur(self):  # pylint: disable=missing-docstring
        return self._semibeat_dur

    @property
    def superbeat_dur(self):  # pylint: disable=missing-docstring
        return self._superbeat_dur

    @property
    def is_compound(self):
        return self._compound

    @property
    def is_triple(self):
        return self._triple

    @property
    def is_duple(self):
        return not self._triple

    def __call__(self, time):
        return self._weight(time)

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

    def _weight(self, time):
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

    def _between(self, length, onset, release, inclusive=False):
        onset, release = self._get_bounds(onset, release)
        time = self._first_after(onset, length)
        out = []
        op = operator.le if inclusive else operator.lt
        while op(time, release):
            out.append({"onset": time, "weight": self(time)})
            time += length
        return out

    def beats_between(self, *args, **kwargs):
        return self._between(self._beat_dur, *args, **kwargs)

    def semibeats_between(self, *args, **kwargs):
        return self._between(self._semibeat_dur, *args, **kwargs)

    def superbeats_between(self, *args, **kwargs):
        return self._between(self._superbeat_dur, *args, **kwargs)


class RhythmFetcher(TimeClass):
    # What this class should ideally do is take a time signature (best way of
    # representing this?) and then return various "rhythms" (e.g., offbeat 8ths;
    # trochees; dactyls; etc.) adapted to that time signature

    def __init__(self, ts_str: str):
        self.meter = Meter(ts_str)
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
            onset, release, inclusive=inclusive
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
