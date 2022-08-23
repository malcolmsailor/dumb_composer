import math
from dataclasses import dataclass
from functools import cached_property, partial
from numbers import Number
import typing as t
import textwrap

from dumb_composer.time_utils import get_min_ioi, get_max_ioi
from dumb_composer.utils.recursion import UndoRecursiveStep

from enum import Enum


class Allow(Enum):
    NO = 1
    YES = 2
    ONLY = 3


class MissingPrefabError(UndoRecursiveStep):
    pass


@dataclass
class PrefabRhythms:
    onsets: t.Sequence[Number]
    metric_strength_str: str
    total_dur: Number
    # if releases is omitted it is generated automagically
    releases: t.Optional[t.Sequence[Number]] = None
    # endpoint_metric_strength_str indicates the metric strengths of the
    #   onset of the rhythm and the onset of the start of the next rhythm
    # E.g., if it goes from beat 1 to beat 1, should be "ss"
    #   if it goes from beat 2 to beat 1, should be "ws"
    endpoint_metric_strength_str: str = "ss"
    allow_suspension: t.Optional[Allow] = None
    allow_preparation: Allow = Allow.NO
    allow_after_tie: t.Optional[Allow] = None

    # we scale the prefab rhythms by factors of 2 to increase the vocabulary.
    # So, e.g., (0.0, 1.0, 3.0 )could also become (0.0, 0.5, 1.5), or
    # (0.0, 2.0, 6.0). We can set limits on the scaling according to the nature
    # of the rhythm.
    min_scaled_ioi: Number = 0.25  # should be a power of 2
    max_scaled_ioi: Number = 4.0  # should be a power of 2

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
        if self.releases is None:
            self.releases = self.onsets[1:] + [self.total_dur]
        assert len(self.metric_strength_str) == len(self.onsets)
        # TODO I think I can remove
        # assert math.log2(self.min_scaled_ioi) % 1 == 0
        # assert math.log2(self.max_scaled_ioi) % 1 == 0

    def __len__(self):
        return len(self.metric_strength_str)

    def matches_criteria(
        self,
        total_dur: Number,
        endpoint_metric_strength_str: str,
        is_suspension: bool = False,
        is_preparation: bool = False,
        is_after_tie: bool = False,
    ):
        if not (
            self.total_dur == total_dur
            and self.endpoint_metric_strength_str
            == endpoint_metric_strength_str
        ):
            return False
        for is_so, allowed in (
            (is_suspension, self.allow_suspension),
            (is_preparation, self.allow_preparation),
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
            onsets=[onset * factor for onset in self.onsets],
            metric_strength_str=self.metric_strength_str,
            total_dur=self.total_dur * factor,
            releases=[release * factor for release in self.releases],
            endpoint_metric_strength_str=self.endpoint_metric_strength_str,
        )

    @cached_property
    def min_ioi(self):
        return get_min_ioi(list(self.onsets) + [self.total_dur])

    @cached_property
    def max_ioi(self):
        return get_max_ioi(list(self.onsets) + [self.total_dur])


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

PREFABS = (
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
)


def scale_prefabs(prefabs: t.Sequence[PrefabRhythms]) -> t.List[PrefabRhythms]:
    out = []
    for prefab_rhythm in prefabs:
        out.append(prefab_rhythm)
        lower_bound = math.floor(
            math.log2(prefab_rhythm.min_ioi)
            - math.log2(prefab_rhythm.min_scaled_ioi)
        )
        for i in range(1, lower_bound + 1):
            out.append(prefab_rhythm.scale(2 ** -(i)))
        upper_bound = math.floor(
            math.log2(prefab_rhythm.max_scaled_ioi)
            - math.log2(prefab_rhythm.max_ioi)
        )
        for i in range(1, upper_bound + 1):
            out.append(prefab_rhythm.scale(2**i))
    return out


PREFABS = scale_prefabs(PREFABS)


class PrefabRhythmDirectory:
    def __init__(self):
        self._memo = {}

    def __call__(
        self,
        total_dur: Number,
        endpoint_metric_strength_str: str = "ss",
        is_suspension: bool = False,
        is_preparation: bool = False,
        is_after_tie: bool = False,
    ) -> t.List[PrefabRhythms]:
        tup = (
            total_dur,
            endpoint_metric_strength_str,
            is_suspension,
            is_preparation,
            is_after_tie,
        )
        if tup in self._memo:
            return self._memo[tup].copy()
        out = [prefab for prefab in PREFABS if prefab.matches_criteria(*tup)]
        if not out:
            raise MissingPrefabError(
                textwrap.dedent(
                    f"""No PrefabRhythms instance matching criteria
                    \ttotal_dur: {total_dur}
                    \tendpoint_metric_strength_str: {endpoint_metric_strength_str}
                    \tis_suspension: {is_suspension}
                    \tis_preparation: {is_preparation}
                    \tis_after_tie: {is_after_tie}
                    """
                )
            )
        self._memo[tup] = out
        return out.copy()
