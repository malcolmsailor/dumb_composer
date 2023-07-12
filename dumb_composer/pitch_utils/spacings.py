import typing as t
from dataclasses import dataclass


@dataclass
class SpacingConstraints:
    max_adjacent_interval: int = 12
    max_total_interval: t.Optional[int] = None
    control_bass_interval: bool = False
    max_bass_interval: t.Optional[int] = None


def validate_spacing(
    spacing: t.Sequence[int],
    max_adjacent_interval: int,
    max_total_interval: t.Optional[int] = None,
    control_bass_interval: bool = False,
    max_bass_interval: t.Optional[int] = None,
) -> bool:
    """
    Args:
        spacing: pitches to evaluate. Should be in sorted order.
        max_adjacent_interval: maximum interval between adjacent voices. Whether this
            argument applies to the bass depends on the `control_bass_interval` and
            `max_bass_interval` arguments.
        max_total_interval: if not None, controls the maximum interval between the outer
            voices.
        control_bass_interval: if True, then we check the interval between the
            bass and the next highest voice. (Otherwise we don't check this interval.)
        max_bass_interval: has no effect if control_bass_interval is False. Otherwise,
            specifies the maximum interval between the bass and the next highest voice.
            If None, we use max_adjacent_interval for this purpose.

    >>> validate_spacing([48, 60, 67, 76], 12)
    True
    >>> validate_spacing([48, 60, 67, 76], 9)
    True
    >>> validate_spacing([48, 60, 67, 76], 9, control_bass_interval=True)
    False
    >>> validate_spacing([48, 60, 67, 76], 9, max_bass_interval=12)
    True
    >>> validate_spacing([48, 60], 9, control_bass_interval=True, max_bass_interval=12)
    True
    >>> validate_spacing([48], 1)
    True
    >>> validate_spacing([48, 60, 67, 76], 9, max_total_interval=24,
    ...                   control_bass_interval=True, max_bass_interval=12)
    False

    """
    if len(spacing) < 2:
        return True

    # TODO: (Malcolm) can we assume that spacing is sorted?
    assert list(spacing) == sorted(spacing)

    if max_total_interval is not None:
        if spacing[-1] - spacing[0] > max_total_interval:
            return False

    if control_bass_interval:
        if max_bass_interval is None:
            max_bass_interval = max_adjacent_interval

        if spacing[1] - spacing[0] > max_bass_interval:
            return False

    for p1, p2 in zip(spacing[1:], spacing[2:]):
        if p2 - p1 > max_adjacent_interval:
            return False

    return True
