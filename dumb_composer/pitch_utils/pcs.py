from collections import Counter
import typing as t

from dumb_composer.pitch_utils.types import Pitch

QuasiPitch = int
PitchClass = int


def get_pc_complement(
    all_pcs: t.Iterable[PitchClass],
    pitches_or_pcs_to_complement: t.Iterable[PitchClass],
    raise_exception: bool = True,
) -> t.List[PitchClass]:
    """
    Note: the notion of complement is a little different than in atonal music theory
    because we allow pcs to occur more than once.

    >>> get_pc_complement([0, 0, 4, 7], [0, 4])
    [0, 7]

    >>> get_pc_complement([0, 0, 4, 7], [60, 72])
    [4, 7]

    If `raise_exception` is True, then any excess pcs in `pitches_or_pcs_to_complement`
    cause a ValueError:
    >>> get_pc_complement([0, 0, 4, 7], [0, 4, 8])
    Traceback (most recent call last):
    ValueError
    >>> get_pc_complement([0, 0, 4, 7], [60, 64, 76])
    Traceback (most recent call last):
    ValueError
    """
    remaining = Counter(p % 12 for p in pitches_or_pcs_to_complement)
    out = []
    for pc in all_pcs:
        if remaining[pc]:
            remaining[pc] -= 1
        else:
            out.append(pc)
    if raise_exception and remaining.total():
        raise ValueError()
    return out


def quasi_pitch_order(
    pcs: t.Sequence[PitchClass], *, tet: int = 12
) -> t.Tuple[QuasiPitch]:
    """
    >>> quasi_pitch_order((0, 4, 7))
    (0, 4, 7)
    >>> quasi_pitch_order((0, 7, 4))
    (0, 7, 16)
    >>> quasi_pitch_order((7, 4, 0))
    (7, 16, 24)
    >>> quasi_pitch_order((1,))
    (1,)
    >>> quasi_pitch_order(())
    ()
    """
    out = list(pcs)  # creates a copy even if pcs is a list
    for i in range(1, len(out)):
        while out[i] < out[i - 1]:
            out[i] += tet
    return tuple(out)


def pitch_class_among_pitches(
    pc: PitchClass, pitches: t.Iterable[Pitch], tet: int = 12
) -> bool:
    """
    >>> pitch_class_among_pitches(4, [60, 67, 76])
    True
    >>> pitch_class_among_pitches(4, [60, 67, 75])
    False
    >>> pitch_class_among_pitches(4, [])
    False
    >>> pitch_class_among_pitches(4, [60, 60, 60])
    False
    >>> pitch_class_among_pitches(4, [60, 64, 76, 88])
    True
    """
    return any(pc == p % tet for p in pitches)
