import typing as t
import math
import warnings
import numpy as np

from dumb_composer.pitch_utils.put_in_range import get_all_in_range


def get_forbidden_intervals(
    starting_pitch: int,
    existing_pitch_pairs: t.Sequence[t.Tuple[int, int]],
    forbidden_parallels: t.Sequence[int],
    forbidden_antiparallels: t.Sequence[int] = (),
) -> t.List[int]:
    """Get intervals (if any) from starting_pitch that would cause forbidden
    parallels with existing_pitch_pairs.

    >>> get_forbidden_intervals(60, [(53, 52), (72, 76)], [0,7])
    [-1, 4]
    >>> sorted(get_forbidden_intervals(60, [(53, 52), (72, 76)], [0,7],
    ...     forbidden_antiparallels=[0, 7]))
    [-8, -1, 4, 11]
    """
    out = []
    for p1, p2 in existing_pitch_pairs:
        harmonic_interval = (starting_pitch - p1) % 12
        melodic_interval = p2 - p1
        if harmonic_interval in forbidden_parallels:
            out.append(melodic_interval)
        if harmonic_interval in forbidden_antiparallels:
            out.append(melodic_interval + -12 * np.sign(melodic_interval))
    return out


def interval_finder(
    starting_pitch: int,
    eligible_pcs: t.Sequence[int],
    min_pitch: int,
    max_pitch: int,
    max_interval: t.Optional[int] = None,
    forbidden_intervals: t.Optional[t.Sequence[int]] = None,
) -> t.List[int]:
    """
    >>> sorted(interval_finder(60, eligible_pcs=[0, 4, 7], min_pitch=48, max_pitch=72))
    [-12, -8, -5, 0, 4, 7, 12]

    Note that forbidden_intervals are *not* octave equivalent:
    >>> sorted(interval_finder(60, eligible_pcs=[2, 6, 9], min_pitch=48, max_pitch=72,
    ...     forbidden_intervals=[6]))
    [-10, -6, -3, 2, 9]

    >>> sorted(interval_finder(60, eligible_pcs=[2, 6, 9], min_pitch=48, max_pitch=72,
    ...     forbidden_intervals=[-6, 6]))
    [-10, -3, 2, 9]
    """
    if max_interval is not None:
        min_pitch = max(min_pitch, starting_pitch - max_interval)
        max_pitch = min(max_pitch, starting_pitch + max_interval)
    eligible_pitches = get_all_in_range(eligible_pcs, min_pitch, max_pitch)
    intervals = [eligible_p - starting_pitch for eligible_p in eligible_pitches]
    if forbidden_intervals:
        return [
            interval
            for interval in intervals
            if not interval in forbidden_intervals
        ]
    return intervals


def get_relative_chord_factors(
    chord_factor: int, chord_intervals: t.Tuple[int], scale_card: int
) -> t.Tuple[int]:
    """Given a chord factor expressed as a generic interval above the root,
    return a tuple of generic intervals to the other factors of the chord,
    both up and down.

    >>> get_relative_chord_factors(0, (0, 2, 4), 7)
    (-5, -3, 2, 4)
    >>> get_relative_chord_factors(2, (0, 2, 5), 8)
    (-5, -2, 3, 6)
    """
    up = tuple(
        sorted(
            (other_factor - chord_factor) % scale_card
            for other_factor in chord_intervals
            if other_factor != chord_factor
        )
    )

    down = tuple(f - scale_card for f in up)
    return down + up
