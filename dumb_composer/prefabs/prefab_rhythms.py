import math
import textwrap
import typing as t
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from numbers import Number

from dumb_composer.pitch_utils.types import TIME_TYPE, TimeStamp
from dumb_composer.shared_classes import Allow
from dumb_composer.time_utils import get_max_ioi, get_min_ioi
from dumb_composer.utils.recursion import UndoRecursiveStep


class MissingPrefabError(UndoRecursiveStep):
    pass


MIN_DEFAULT_RHYTHM_BEFORE_REST = 1 / 2


class PrefabBase:
    pass


class PrefabRhythmBase(PrefabBase):
    onsets: t.Sequence[TimeStamp]
    releases: t.Sequence[TimeStamp] | None


@dataclass(frozen=True, init=False)
class SingletonRhythm(PrefabRhythmBase):
    onsets: t.Tuple[TIME_TYPE] = (TIME_TYPE(0),)
    releases: t.Sequence[TimeStamp] | None = None
    metric_strength_str: str = "_"
    allow_suspension: Allow = Allow.NO
    allow_preparation: Allow = Allow.NO
    allow_after_tie: bool = False
    allow_next_to_start_with_rest: Allow = Allow.NO

    def matches_criteria(self, *args, **kwargs):
        return True


@dataclass
class PrefabRhythms(PrefabRhythmBase):
    onsets: t.List[TimeStamp]
    metric_strength_str: str
    total_dur: TIME_TYPE
    # if releases is omitted it is generated automagically
    releases: t.Optional[t.Sequence[TimeStamp]] = None
    # endpoint_metric_strength_str indicates the metric strengths of the
    #   onset of the rhythm and the onset of the start of the next rhythm
    # E.g., if it goes from beat 1 to beat 1, should be "ss"
    #   if it goes from beat 2 to beat 1, should be "ws"
    endpoint_metric_strength_str: str = "ss"
    allow_suspension: t.Optional[Allow] = None
    allow_preparation: Allow = Allow.NO
    allow_resolution: t.Optional[Allow] = None
    allow_after_tie: t.Optional[Allow] = None

    # if allow_ext_to_start_with_rest is None, we set it according to the
    #   following heuristic:
    #       - if the rhythm ends with a rest, set to Allow.YES
    #       - else if last onset is <= 1/4 the length of the rhythm, or less
    #           than MIN_DEFAULT_RHYTHM_BEFORE_REST set to allow.NO
    #       - else, set to allow.YES
    allow_next_to_start_with_rest: t.Optional[Allow] = None

    # we scale the prefab rhythms by factors of 2 to increase the vocabulary.
    # So, e.g., (0.0, 1.0, 3.0 )could also become (0.0, 0.5, 1.5), or
    # (0.0, 2.0, 6.0). We can set limits on the scaling according to the nature
    # of the rhythm.
    min_scaled_ioi: Number = 0.25  # type:ignore
    max_scaled_ioi: Number = 4.0  # type:ignore

    def __post_init__(self):
        if self.allow_after_tie is None:
            if not self.onsets[0] == 0.0:
                self.allow_after_tie = Allow.NO
            else:
                self.allow_after_tie = Allow.YES
        if self.allow_suspension is None:
            if not self.onsets[0] == 0.0:
                self.allow_suspension = Allow.NO
            else:
                self.allow_suspension = Allow.YES
        if self.allow_resolution is None:
            if not self.onsets[0] == 0.0:
                self.allow_resolution = Allow.NO
            else:
                self.allow_resolution = Allow.YES
        if self.releases is None:
            self.releases = self.onsets[1:] + [self.total_dur]
        if self.allow_next_to_start_with_rest is None:
            if self.releases[-1] < self.total_dur:  # type:ignore
                self.allow_next_to_start_with_rest = Allow.YES
            else:
                last_dur = self.total_dur - self.onsets[-1]  # type:ignore
                if (
                    last_dur < MIN_DEFAULT_RHYTHM_BEFORE_REST
                    or last_dur / self.total_dur <= 1 / 4
                ):
                    self.allow_next_to_start_with_rest = Allow.NO
                else:
                    self.allow_next_to_start_with_rest = Allow.YES
        assert len(self.metric_strength_str) == len(self.onsets)

    def __len__(self):
        return len(self.metric_strength_str)

    def matches_criteria(
        self,
        total_dur: Number,
        endpoint_metric_strength_str: str,
        is_suspension: bool = False,
        is_preparation: bool = False,
        is_resolution: bool = False,
        is_after_tie: bool = False,
        start_with_rest: Allow = Allow.YES,
    ):
        if not (
            self.total_dur == total_dur
            and self.endpoint_metric_strength_str == endpoint_metric_strength_str
        ):
            return False
        if start_with_rest is Allow.NO and self.onsets[0] != 0:
            return False
        elif start_with_rest is Allow.ONLY and self.onsets[0] == 0:
            return False
        for is_so, allowed in (
            (is_suspension, self.allow_suspension),
            (is_preparation, self.allow_preparation),
            (is_resolution, self.allow_resolution),
            (is_after_tie, self.allow_after_tie),
        ):
            if is_so and allowed == Allow.NO:
                return False
            if not is_so and allowed == Allow.ONLY:
                return False
        return True

    def scale(self, factor: Number):
        """Return a copy of self with rhythmic values scaled by factor.

        min_scaled_ioi and max_scaled_ioi are not changed.
        """
        return self.__class__(
            onsets=[onset * factor for onset in self.onsets],  # type:ignore
            metric_strength_str=self.metric_strength_str,
            total_dur=self.total_dur * factor,  # type:ignore
            releases=[release * factor for release in self.releases],  # type:ignore
            endpoint_metric_strength_str=self.endpoint_metric_strength_str,
        )

    @cached_property
    def min_ioi(self):
        return get_min_ioi(list(self.onsets) + [self.total_dur])

    @cached_property
    def max_ioi(self):
        return get_max_ioi(list(self.onsets) + [self.total_dur])

    def __hash__(self):
        """We want each PrefabRhythms instance to be considered unique so we
        just return the id of the instance."""
        return id(self)

    def __str__(self):
        return str(
            [
                f"{onset}-{release}"
                for (onset, release) in zip(self.onsets, self.releases)  # type:ignore
            ]
        )


def match_metric_strength_strs(
    metric_strength_str1: str, metric_strength_str2: str
) -> bool:
    """Two "metric strength strings" match at each position if each character
    is the same or if one of the strings has a wildcard '_'.

    I am using "s" for strong and "w" for weak but this is not enforced; any
    characters are ok; the only character treated specially is '_'.

    >>> match_metric_strength_strs("swsw", "swsw")
    True
    >>> match_metric_strength_strs("swsw", "swws")
    False
    >>> match_metric_strength_strs("s___", "swsw")
    True

    If the two strings are of different length they don't match.

    >>> match_metric_strength_strs("swsw", "sws")
    False
    """
    if len(metric_strength_str1) != len(metric_strength_str2):
        return False
    for ch1, ch2 in zip(metric_strength_str1, metric_strength_str2):
        if ch1 != ch2 and "_" not in (ch1, ch2):
            return False
    return True


PR = PrefabRhythms

PR4 = partial(PR, total_dur=4)
PR3 = partial(PR, total_dur=3)
PR2 = partial(PR, total_dur=2)

PREFABS = [
    PR4([0.0, 1.5, 2.0, 2.5, 3.0], "swsws", allow_preparation=Allow.YES),
    PR4([0.0, 1.0, 2.0, 3.0], "swsw"),
    PR4([0.0, 1.5, 2.0, 3.0], "swsw"),
    PR4([0.0, 1.0, 2.0], "s_s", allow_preparation=Allow.YES),
    PR3([0.0, 1.0, 2.0], "s__", allow_preparation=Allow.YES),
    PR3([0.0, 1.0, 1.5, 2.0], "s___"),
    PR3([0.0, 1.5, 2.0, 2.5], "s___"),
    PR2([0.0, 1.5], "sw"),
    PR2([1.0, 1.5], "__"),
    PR2([0.0, 1.0, 1.5], "s__"),
    PR2([0.0, 1.0], "__"),
    PR2([0.0], "_"),
]


def scale_prefabs(prefabs: t.Sequence[PrefabRhythms]) -> t.List[PrefabRhythms]:
    out = []
    for prefab_rhythm in prefabs:
        out.append(prefab_rhythm)
        lower_bound = math.floor(
            math.log2(prefab_rhythm.min_ioi)  # type:ignore
            - math.log2(prefab_rhythm.min_scaled_ioi)  # type:ignore
        )
        for i in range(1, lower_bound + 1):
            out.append(prefab_rhythm.scale(2 ** -(i)))
        upper_bound = math.floor(
            math.log2(prefab_rhythm.max_scaled_ioi)  # type:ignore
            - math.log2(prefab_rhythm.max_ioi)  # type:ignore
        )
        for i in range(1, upper_bound + 1):
            out.append(prefab_rhythm.scale(2**i))
    return out


PREFABS = scale_prefabs(PREFABS)


class PrefabRhythmDirectory:
    """A directory of prefab rhythms.

    >>> prefab_rhythm_dir = PrefabRhythmDirectory()
    >>> eight_beat_rhythms = prefab_rhythm_dir(8.0)
    >>> len(eight_beat_rhythms)  # doctest: +SKIP
    7
    >>> eight_beat_rhythms[0]  # doctest: +SKIP
    PrefabRhythms(
        onsets=[0.0, 3.0, 4.0, 5.0, 6.0],
        metric_strength_str='swsws', total_dur=8, releases=[3.0, 4.0, 5.0, 6.0, 8],
        endpoint_metric_strength_str='ss', allow_suspension=<Allow.YES: 2>,
        allow_preparation=<Allow.NO: 1>, allow_resolution=<Allow.YES: 2>,
        allow_after_tie=<Allow.YES: 2>, allow_next_to_start_with_rest=<Allow.NO: 1>,
        min_scaled_ioi=0.25, max_scaled_ioi=4.0)
    """

    def __init__(self, allow_singleton_rhythm: bool = True):
        self._memo = {}
        self._singleton = SingletonRhythm() if allow_singleton_rhythm else None

    def __call__(
        self,
        total_dur: Number,
        endpoint_metric_strength_str: str = "ss",
        is_suspension: bool = False,
        is_preparation: bool = False,
        is_resolution: bool = False,
        is_after_tie: bool = False,
        start_with_rest: Allow = Allow.YES,
    ) -> t.Union[t.List[PrefabRhythms], t.List[SingletonRhythm]]:
        tup = (
            total_dur,
            endpoint_metric_strength_str,
            is_suspension,
            is_preparation,
            is_resolution,
            is_after_tie,
            start_with_rest,
        )
        if tup in self._memo:
            return self._memo[tup].copy()
        out = [prefab for prefab in PREFABS if prefab.matches_criteria(*tup)]
        if not out:
            if self._singleton is not None:
                # TODO store missing rhythms for reference
                out = [self._singleton]
            else:
                raise MissingPrefabError(
                    textwrap.dedent(
                        f"""No PrefabRhythms instance matching criteria
                        \ttotal_dur: {total_dur}
                        \tendpoint_metric_strength_str: {endpoint_metric_strength_str}
                        \tis_suspension: {is_suspension}
                        \tis_preparation: {is_preparation}
                        \tis_resolution: {is_resolution}
                        \tis_after_tie: {is_after_tie}
                        \tstart_with_rest: {start_with_rest}"""
                    )
                )
        self._memo[tup] = out
        return out.copy()
