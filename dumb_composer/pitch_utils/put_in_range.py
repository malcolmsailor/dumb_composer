import math
import random
import typing as t
from functools import partial, reduce

from dumb_composer.pitch_utils.types import Pitch, PitchClass, PitchOrPitchClass
from dumb_composer.utils.nested import nested

# TODO: (Malcolm) maybe consolidate these functions between libraries


@nested()
def put_in_range(p, low=None, high=None, tet=12, fail_silently: bool = False):
    """Used by voice_lead_pitches().

    >>> put_in_range(0, low=58, high=74)
    60

    Moves pitch as little as possible while keeping it within the specified
    range:

    >>> put_in_range(72, low=32, high=62)
    60
    >>> put_in_range(48, low=58, high=100)
    60

    Raises an exception if the pitch-class isn't found between the bounds,
    unless fail_silently is True, in which case it returns a pitch below the
    lower bound:

    >>> put_in_range(60, low=49, high=59)
    Traceback (most recent call last):
        raise ValueError(
    ValueError: pitch-class 0 does not occur between low=49 and high=59

    >>> put_in_range(60, low=49, high=59, fail_silently=True)
    48
    """
    if low is not None:
        below = low - p
        if below > 0:
            octaves_below = math.ceil((below) / tet)
            p += octaves_below * tet
    if high is not None:
        above = p - high
        if above > 0:
            octaves_above = math.ceil(above / tet)
            p -= octaves_above * tet
    if not fail_silently and low is not None:
        if p < low:
            raise ValueError(
                f"pitch-class {p % 12} does not occur between "
                f"low={low} and high={high}"
            )
    return p


def get_all_in_range(
    p: t.Sequence[PitchOrPitchClass] | PitchOrPitchClass,
    low: Pitch,
    high: Pitch,
    steps_per_octave: int = 12,
    sorted: bool = False,
    shuffled: bool = False,
) -> t.List[int]:
    """Bounds are inclusive.

    >>> get_all_in_range(60, low=58, high=72)
    [60, 72]
    >>> get_all_in_range(60, low=58, high=59)
    []
    >>> get_all_in_range(58, low=58, high=85)
    [58, 70, 82]
    >>> get_all_in_range(58, low=58, high=85, shuffled=True)  # doctest: +SKIP
    [70, 82, 58]

    If a single pitch-class is passed and shuffled=False, the output is always
    sorted. But with multiple pitch-classes we need to add sorted=True if we want
    the output to be in order:
    >>> get_all_in_range([58, 60], low=58, high=83)
    [58, 70, 82, 60, 72]
    >>> get_all_in_range([58, 60], low=58, high=83, sorted=True)
    [58, 60, 70, 72, 82]
    >>> get_all_in_range([58, 60], low=58, high=83, shuffled=True)  # doctest: +SKIP
    [60, 72, 70, 82, 58]

    This function can also be used on scale degrees or scale indices if
    `steps_per_octave` is set appropriately.
    >>> get_all_in_range(  # scalar thirds between 3rd and 6th octaves
    ...     2, low=21, high=42, steps_per_octave=7
    ... )
    [23, 30, 37]
    >>> get_all_in_range(  # tonic triad between 3rd and 5th octaves
    ...     [0, 2, 4], low=28, high=42, steps_per_octave=7, sorted=True
    ... )
    [28, 30, 32, 35, 37, 39, 42]
    """
    if not isinstance(p, int):
        if not p:
            return []
        out = reduce(
            lambda x, y: x + y,
            [get_all_in_range(pp, low, high, steps_per_octave) for pp in p],
        )
        if shuffled:
            random.shuffle(out)
        elif sorted:
            out.sort()
        return out
    pc = p % steps_per_octave
    low_octave, low_pc = divmod(low, steps_per_octave)
    low_octave += pc < low_pc
    high_octave, high_pc = divmod(high, steps_per_octave)
    high_octave -= pc > high_pc
    octaves = range(low_octave, high_octave + 1)
    if shuffled:
        octaves = random.sample(octaves, len(octaves))
    return [pc + octave * steps_per_octave for octave in octaves]


# TODO: (Malcolm 2023-07-13) do we actually need this?
def yield_all_in_range(
    p: t.Sequence[PitchOrPitchClass] | PitchOrPitchClass,
    low: Pitch,
    high: Pitch,
    tet: int = 12,
) -> t.Iterable[Pitch]:
    """
    >>> list(yield_all_in_range(60, low=58, high=72))
    [60, 72]
    >>> list(yield_all_in_range(60, low=58, high=59))
    []
    >>> list(yield_all_in_range(58, low=58, high=85))
    [58, 70, 82]

    Note: sorted is not yet implemented
    >>> list(yield_all_in_range([58, 60], low=58, high=83))
    [58, 70, 82, 60, 72]
    """
    if not isinstance(p, int):
        for x in p:
            yield from yield_all_in_range(x, low, high, tet)
        return
    pc = p % tet
    low_octave, low_pc = divmod(low, tet)
    low_octave += pc < low_pc
    high_octave, high_pc = divmod(high, tet)
    high_octave -= pc > high_pc
    yield from (pc + octave * tet for octave in range(low_octave, high_octave + 1))
