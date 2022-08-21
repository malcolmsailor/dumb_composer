from functools import partial, reduce
import math
import typing as t
from dumb_composer.utils.nested import nested


@nested()
def put_in_range(p, low=None, high=None, tet=12):
    """
    >>> put_in_range(0, low=58, high=74)
    60
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
    return p


def get_all_in_range(
    p: t.Union[t.List[int], int],
    low: int,
    high: int,
    tet: int = 12,
    sorted: bool = False,
) -> t.List[int]:
    """Bounds are inclusive.

    >>> get_all_in_range(60, low=58, high=72)
    [60, 72]
    >>> get_all_in_range(60, low=58, high=59)
    []
    >>> get_all_in_range(58, low=58, high=85)
    [58, 70, 82]
    >>> get_all_in_range([58, 60], low=58, high=83, sorted=True)
    [58, 60, 70, 72, 82]
    """
    if not isinstance(p, int):
        if not p:
            return []
        out = reduce(
            lambda x, y: x + y,
            [get_all_in_range(pp, low, high, tet) for pp in p],
        )
        if sorted:
            out.sort()
        return out
    pc = p % tet
    low_octave, low_pc = divmod(low, tet)
    low_octave += pc < low_pc
    high_octave, high_pc = divmod(high, tet)
    high_octave -= pc > high_pc
    return [pc + octave * tet for octave in range(low_octave, high_octave + 1)]


# TODO put_in_range with smoothing [what did I mean by this?]
