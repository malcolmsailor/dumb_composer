from dataclasses import dataclass
from functools import partial
from numbers import Number
import typing as t
import textwrap

from dumb_composer.utils.recursion import UndoRecursiveStep


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

    def __post_init__(self):
        if self.releases is None:
            self.releases = self.onsets[1:] + [self.total_dur]
        assert len(self.metric_strength_str) == len(self.onsets)

    def __len__(self):
        return len(self.metric_strength_str)

    def matches_criteria(
        self, total_dur: Number, endpoint_metric_strength_str: str
    ):
        return (
            self.total_dur == total_dur
            and self.endpoint_metric_strength_str
            == endpoint_metric_strength_str
        )


def match_metric_strength_strs(
    metric_strength_str1: str, metric_strength_str2: str
) -> bool:
    """Two "metric strength strings" match at each position if each character
    is the same or if one of the strings has a wildcard '_'.

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

PREFABS = (
    PR4([0.0, 1.0, 2.0, 3.0], "swsw"),
    PR4([0.0, 1.5, 2.0, 3.0], "swsw"),
    PR3([0.0, 1.0, 2.0], "s__"),
    PR3([0.0, 1.0, 1.5, 2.0], "s___"),
    PR3([0.0, 1.5, 2.0, 2.5], "s___"),
)


class PrefabRhythmDirectory:
    def __init__(self):
        self._memo = {}

    def __call__(
        self, total_dur: Number, endpoint_metric_strength_str: str = "ss"
    ) -> t.List[PrefabRhythms]:
        tup = (total_dur, endpoint_metric_strength_str)
        if tup in self._memo:
            return self._memo[tup].copy()
        out = [prefab for prefab in PREFABS if prefab.matches_criteria(*tup)]
        if not out:
            raise MissingPrefabError(
                textwrap.dedent(
                    f"""No PrefabRhythms instance matching criteria
                    \ttotal_dur: {total_dur}
                    \tendpoint_metric_strength_str: {endpoint_metric_strength_str}
                    """
                )
            )
        self._memo[tup] = out
        return out.copy()
