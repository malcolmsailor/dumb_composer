import math
import typing as t
from numbers import Number

import numpy as np


def get_onsets_within_duration(
    start: Number,
    stop: Number,
    grid: Number,
    include_start: bool = True,
    include_stop: bool = False,
) -> t.List[Number]:
    """Inclusive of start, exclusive of stop.
    >>> get_onsets_within_duration(1.6, 3.0, 0.25)
    [1.75, 2.0, 2.25, 2.5, 2.75]
    """
    if include_start:
        initial_i = math.ceil(start / grid)
    else:
        initial_i = math.floor(start / grid) + 1
    final_i = math.ceil(stop / grid)
    if include_stop:
        final_i += 1
    if final_i <= initial_i:
        return []
    return [i * grid for i in range(initial_i, final_i)]


def get_barline_times_within_duration(
    start: Number,
    stop: Number,
    ts_dur: Number,
    include_start: bool = True,
    include_stop: bool = False,
) -> t.List[Number]:
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
    return get_onsets_within_duration(
        start, stop, ts_dur, include_start, include_stop
    )


def get_onset_closest_to_middle_of_duration(
    onsets: t.Sequence[Number], start: Number, stop: Number
) -> Number:
    """
    >>> get_onset_closest_to_middle_of_duration([0.0, 4.0, 8.0], -4.0, 12.0)
    4.0
    >>> get_onset_closest_to_middle_of_duration([3.75, 4.0, 4.01], 4.0, 4.0)
    4.0
    >>> get_onset_closest_to_middle_of_duration([3.75, 4.0, 4.01], 4.0, 4.0151)
    4.01
    """
    midpoint = (start + stop) / 2
    manhattan_distance = [abs(onset - midpoint) for onset in onsets]
    return onsets[np.argmin(manhattan_distance)]


def get_max_ioi(onsets: t.Sequence[Number]) -> Number:
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
    return max(y - x for x, y in zip(onsets[:-1], onsets[1:]))


def get_min_ioi(onsets: t.Sequence[Number]) -> Number:
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
    return min(y - x for x, y in zip(onsets[:-1], onsets[1:]))
