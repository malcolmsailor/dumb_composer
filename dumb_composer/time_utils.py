import math
import typing as t
from numbers import Number

import numpy as np

from dumb_composer.constants import TIME_TYPE


def get_onsets_within_duration(
    start: TIME_TYPE,
    stop: TIME_TYPE,
    grid: TIME_TYPE,
    include_start: bool = True,
    include_stop: bool = False,
) -> t.List[TIME_TYPE]:
    """Inclusive of start, exclusive of stop.
    >>> get_onsets_within_duration(1.6, 3.0, 0.25)
    [1.75, 2.0, 2.25, 2.5, 2.75]
    """
    if include_start:
        initial_i = math.ceil(start / grid)  # type:ignore
    else:
        initial_i = math.floor(start / grid) + 1  # type:ignore
    final_i = math.ceil(stop / grid)  # type:ignore
    if include_stop:
        final_i += 1
    if final_i <= initial_i:
        return []
    return [i * grid for i in range(initial_i, final_i)]  # type:ignore


def get_barline_times_within_duration(
    start: TIME_TYPE,
    stop: TIME_TYPE,
    ts_dur: TIME_TYPE,
    include_start: bool = True,
    include_stop: bool = False,
) -> t.List[TIME_TYPE]:
    """
    >>> get_barline_times_within_duration(0.0, 16.0, 4.0)
    [0.0, 4.0, 8.0, 12.0]
    >>> get_barline_times_within_duration(0.0, 16.0, 4.0, include_start=False)
    [4.0, 8.0, 12.0]
    >>> get_barline_times_within_duration(0.0, 16.0, 4.0, include_stop=True)
    [0.0, 4.0, 8.0, 12.0, 16.0]
    >>> get_barline_times_within_duration(0.5, 4.0, 4.0)
    []
    >>> get_barline_times_within_duration(0.5, 4.0, 3.0)
    [3.0]
    >>> get_barline_times_within_duration(0.5, 4.0, 3.0, include_start=False)
    [3.0]
    >>> get_barline_times_within_duration(0.0, 0.0, 3.0)
    []
    >>> get_barline_times_within_duration(0.0, 15.9, 4.0)
    [0.0, 4.0, 8.0, 12.0]
    """
    return get_onsets_within_duration(start, stop, ts_dur, include_start, include_stop)


def get_onset_closest_to_middle_of_duration(
    onsets: t.Sequence[TIME_TYPE], start: TIME_TYPE, stop: TIME_TYPE
) -> TIME_TYPE:
    """
    >>> get_onset_closest_to_middle_of_duration([0.0, 4.0, 8.0], -4.0, 12.0)
    4.0
    >>> get_onset_closest_to_middle_of_duration([3.75, 4.0, 4.01], 4.0, 4.0)
    4.0
    >>> get_onset_closest_to_middle_of_duration([3.75, 4.0, 4.01], 4.0, 4.0151)
    4.01
    """
    midpoint = (start + stop) / 2  # type:ignore
    manhattan_distance = [abs(onset - midpoint) for onset in onsets]
    arg_min = np.argmin(manhattan_distance)  # type:ignore
    return onsets[arg_min]


def get_max_ioi(onsets: t.Sequence[TIME_TYPE]) -> TIME_TYPE:
    """
    Unlikely to give right answer if onsets are not sorted.

    >>> get_max_ioi([0.0, 0.25, 0.75, 1.0])
    0.5
    >>> get_max_ioi([0.0])
    Traceback (most recent call last):
    ValueError: There must be at least two onsets
    """
    if len(onsets) < 2:
        raise ValueError("There must be at least two onsets")
    return max(y - x for x, y in zip(onsets[:-1], onsets[1:]))  # type:ignore


def get_min_ioi(onsets: t.Sequence[TIME_TYPE]) -> TIME_TYPE:
    """
    Unlikely to give right answer if onsets are not sorted.

    >>> get_min_ioi([0.0, 0.25, 0.75, 1.0])
    0.25
    >>> get_min_ioi([0.0])
    Traceback (most recent call last):
    ValueError: There must be at least two onsets
    """
    if len(onsets) < 2:
        raise ValueError("There must be at least two onsets")
    return min(y - x for x, y in zip(onsets[:-1], onsets[1:]))  # type:ignore
