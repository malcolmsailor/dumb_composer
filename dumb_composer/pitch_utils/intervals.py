import typing as t
import math
import warnings
import numpy as np

from dumb_composer.pitch_utils.put_in_range import get_all_in_range


class IntervalQuerier:
    def __init__(self):
        self._pitches_contain_ic_memo = {}
        self._pc_can_be_omitted_memo = {}

    def pitches_contain_ic(
        self, pitches: t.FrozenSet[int], interval: int
    ) -> bool:
        """
        >>> iq = IntervalQuerier()
        >>> iq.pitches_contain_ic([60, 64, 67], 4)
        True
        >>> iq.pitches_contain_ic([60, 64, 67], 8)
        True

        >>> all(iq.pitches_contain_ic([60, 64, 67, 70], i)
        ...     for i in (2, 3, 4, 5, 6, 7, 8, 9, 10))
        True

        >>> not any(iq.pitches_contain_ic([60, 64, 67, 70], i)
        ...         for i in (1, 11))
        True
        """
        # TODO I don't think it's necessary to cast to frozenset; maybe
        #   just tuple instead?
        if not isinstance(pitches, frozenset):
            pitches = frozenset(pitches)
        if (pitches, interval) in self._pitches_contain_ic_memo:
            return self._pitches_contain_ic_memo[(pitches, interval)]
        pitches_list = list(pitches)
        for i, p1 in enumerate(pitches_list[:-1]):
            for p2 in pitches_list[i + 1 :]:
                if interval in ((p1 - p2) % 12, (p2 - p1) % 12):
                    self._pitches_contain_ic_memo[pitches, interval] = True
                    return True
        self._pitches_contain_ic_memo[pitches, interval] = False
        return False

    def pc_can_be_omitted(
        self, pc: int, existing_pitches: t.Tuple[int]
    ) -> bool:
        """
        Returns true if
        1. the pc is not in the bass (i.e., the first item of existing_pitches)
        2. there is already an imperfect consonance or dissonance among
            existing pitches or if this pc would not create an imperfect
            consonance or a dissonance if added to the existing pitches.

        >>> iq = IntervalQuerier()
        >>> iq.pc_can_be_omitted(7, (60, 64))
        True

        >>> iq.pc_can_be_omitted(7, (60, 67))
        True

        >>> iq.pc_can_be_omitted(4, (60, 67))
        False
        """

        if (pc, existing_pitches) in self._pc_can_be_omitted_memo:
            return self._pc_can_be_omitted_memo[pc, existing_pitches]
        if pc == existing_pitches[0] % 12:
            self._pc_can_be_omitted_memo[pc, existing_pitches] = False
            return False
        for ic in (1, 2, 3, 4, 6):
            if self.pitches_contain_ic(existing_pitches, ic):
                self._pc_can_be_omitted_memo[pc, existing_pitches] = True
                return True
        for ic in (1, 2, 3, 4, 6):
            # There's probably a faster way of evaluating this condition
            #   based on the observation that the only way of adding a
            #   pitch-class to two existing pitches without creaing an
            #   imperfect consonance or dissonance is if the two existing
            #   pitches are a perfect consonance and the new pitch doubles
            #   one of them.
            if self.pitches_contain_ic(existing_pitches + (pc,), ic):
                self._pc_can_be_omitted_memo[pc, existing_pitches] = False
                return False
        self._pc_can_be_omitted_memo[pc, existing_pitches] = True
        return True


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
    allow_steps_outside_of_range: bool = False,
) -> t.List[int]:
    """
    >>> sorted(interval_finder(60, eligible_pcs=[0, 4, 7],
    ...        min_pitch=48, max_pitch=72))
    [-12, -8, -5, 0, 4, 7, 12]

    Note that forbidden_intervals are *not* octave equivalent:
    >>> sorted(interval_finder(60, eligible_pcs=[2, 6, 9],
    ...        min_pitch=48, max_pitch=72,
    ...        forbidden_intervals=[6]))
    [-10, -6, -3, 2, 9]

    >>> sorted(interval_finder(60, eligible_pcs=[2, 6, 9],
    ...        min_pitch=48, max_pitch=72,
    ...        forbidden_intervals=[-6, 6]))
    [-10, -3, 2, 9]

    >>> sorted(interval_finder(60, eligible_pcs=[2, 10],
    ...        min_pitch=59, max_pitch=61, allow_steps_outside_of_range=True))
    [-2, 2]
    """
    if max_interval is not None:
        min_pitch = max(min_pitch, starting_pitch - max_interval)
        max_pitch = min(max_pitch, starting_pitch + max_interval)
    if allow_steps_outside_of_range:
        min_pitch = min(min_pitch, starting_pitch - 2)
        max_pitch = max(max_pitch, starting_pitch + 2)
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
