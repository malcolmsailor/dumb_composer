import typing as t
from itertools import chain, count, cycle, repeat

from dumb_composer.pitch_utils.types import ScalarInterval


def ascending_chord_intervals(
    intervals: t.Sequence[ScalarInterval], n_steps_per_octave: int = 7
) -> t.Iterator[ScalarInterval]:
    """
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([0, 2, 4]))]
    [0, 2, 4, 7, 9, 11, 14, 16]
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([0, 1, 3, 5]))]
    [0, 1, 3, 5, 7, 8, 10, 12]
    >>> [i for _, i in zip(range(8), ascending_chord_intervals([1, 2, 4]))]
    [1, 2, 4, 8, 9, 11, 15, 16]
    """
    card = len(intervals)
    for interval, octave in zip(
        cycle(intervals), chain.from_iterable(repeat(o, card) for o in count())
    ):
        yield interval + octave * n_steps_per_octave


def ascending_chord_intervals_within_range(
    intervals: t.Sequence[ScalarInterval],
    inclusive_endpoint: ScalarInterval,
    n_steps_per_octave: int = 7,
):
    """
    >>> list(ascending_chord_intervals_within_range([0, 2, 4], 0))
    [0]
    >>> list(ascending_chord_intervals_within_range([0, 2, 4], 15))
    [0, 2, 4, 7, 9, 11, 14]
    """
    for x in ascending_chord_intervals(intervals, n_steps_per_octave):
        if x > inclusive_endpoint:
            return
        yield x


def descending_chord_intervals(
    intervals: t.Sequence[ScalarInterval], n_steps_per_octave: int = 7
) -> t.Iterator[ScalarInterval]:
    """
    >>> [i for _, i in zip(range(8), descending_chord_intervals([0, 2, 4]))]
    [0, -3, -5, -7, -10, -12, -14, -17]
    >>> [i for _, i in zip(range(8), descending_chord_intervals([0, 1, 3, 5]))]
    [0, -2, -4, -6, -7, -9, -11, -13]
    >>> [i for _, i in zip(range(8), descending_chord_intervals([1, 2, 4]))]
    [-3, -5, -6, -10, -12, -13, -17, -19]
    """
    if intervals[0] == 0:
        yield 0

    card = len(intervals)
    for interval, octave in zip(
        cycle(reversed(intervals)),
        chain.from_iterable(repeat(o, card) for o in count(start=1)),
    ):
        yield interval - octave * n_steps_per_octave


def descending_chord_intervals_within_range(
    intervals: t.Sequence[ScalarInterval],
    inclusive_endpoint: ScalarInterval,
    n_steps_per_octave: int = 7,
):
    """
    >>> list(descending_chord_intervals_within_range([0, 2, 4], 0))
    [0]
    >>> list(descending_chord_intervals_within_range([0, 2, 4], -15))
    [0, -3, -5, -7, -10, -12, -14]
    """
    for x in descending_chord_intervals(intervals, n_steps_per_octave):
        if x < inclusive_endpoint:
            return
        yield x
