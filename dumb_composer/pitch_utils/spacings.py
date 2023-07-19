import random
import typing as t
from dataclasses import dataclass

from dumb_composer.constants import DEFAULT_BASS_RANGE, DEFAULT_MEL_RANGE
from dumb_composer.pitch_utils.put_in_range import get_all_in_range, yield_all_in_range
from dumb_composer.pitch_utils.types import Pitch, PitchClass

DEFAULT_RANGE_MIN_PITCH = 36
DEFAULT_RANGE_MAX_PITCH = 84


class RangeConstraints:
    def __init__(
        self,
        min_pitch: Pitch = DEFAULT_RANGE_MIN_PITCH,
        max_pitch: Pitch = DEFAULT_RANGE_MAX_PITCH,
        min_bass_pitch: Pitch | None = DEFAULT_BASS_RANGE[0],
        max_bass_pitch: Pitch | None = DEFAULT_BASS_RANGE[1],
        min_melody_pitch: Pitch | None = DEFAULT_MEL_RANGE[0],
        max_melody_pitch: Pitch | None = DEFAULT_MEL_RANGE[1],
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_bass_pitch = min_pitch if min_bass_pitch is None else min_bass_pitch
        self.max_bass_pitch = max_pitch if max_bass_pitch is None else max_bass_pitch
        self.min_melody_pitch = (
            min_pitch if min_melody_pitch is None else min_melody_pitch
        )
        self.max_melody_pitch = (
            max_pitch if max_melody_pitch is None else max_melody_pitch
        )


@dataclass
class SpacingConstraints:
    """
    Terminology:
        bass interval: interval between bass (0th pitch) and next voice (1st pitch)
        melody interval: interval between second-last voice and last voice, if there
            are at least three voices.

    Args:
        max_adjacent_interval: maximum interval between adjacent voices. By default
            applies to melody interval but not bass interval. See below. Default 12.

        >>> validate_spacing(
        ...     [48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=9)
        ... )
        True
        >>> validate_spacing(
        ...     [48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=7)
        ... )
        False

        min_adjacent_interval: minimum interval between adjacent voices. To allow
            crossings, set to a negative number. Default 0.

        >>> validate_spacing([52, 60, 60, 67], SpacingConstraints())
        True
        >>> validate_spacing(
        ...     [52, 60, 60, 67], SpacingConstraints(min_adjacent_interval=2)
        ... )
        False
        >>> validate_spacing([52, 60, 58, 67], SpacingConstraints())
        False
        >>> validate_spacing(
        ...     [52, 60, 58, 67], SpacingConstraints(min_adjacent_interval=-5)
        ... )
        True

        max_total_interval: if provided, controls the maximum interval between the outer
            voices.

        >>> validate_spacing(
        ...     [48, 60, 67, 76], SpacingConstraints(max_total_interval=24)
        ... )
        False

        control_bass_interval: if True (default: False), then we check the interval between the
            bass and the next highest voice. (Otherwise we don't check this interval.)
        max_bass_interval: has no effect if control_bass_interval is False. Otherwise,
            specifies the maximum interval between the bass and the next highest voice.
            If None, we use max_adjacent_interval for this purpose.

        >>> validate_spacing(
        ...     [48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=9)
        ... )
        True
        >>> validate_spacing(
        ...     [48, 60, 67, 76],
        ...     SpacingConstraints(max_adjacent_interval=9, control_bass_interval=True),
        ... )
        False
        >>> validate_spacing(
        ...     [48, 60, 67, 76],
        ...     SpacingConstraints(
        ...         max_adjacent_interval=9,
        ...         control_bass_interval=True,
        ...         max_bass_interval=12,
        ...     ),
        ... )
        True

        avoid_bass_crossing: if True (default), then bass crossings are avoided even if
            control_bass_interval is False.

        >>> validate_spacing([60, 58, 67, 76], SpacingConstraints())
        False
        >>> validate_spacing(
        ...     [60, 58, 67, 76], SpacingConstraints(avoid_bass_crossing=False)
        ... )
        True

        min_bass_interval: has no effect if control_bass_interval is False. Otherwise,
            specifies the minimum interval between the bass and the next highest voice.
            If avoid_bass_crossing is True, will be set to 0 if < 0.

        >>> validate_spacing(
        ...     [58, 60, 67, 76],
        ...     SpacingConstraints(control_bass_interval=True, min_bass_interval=3),
        ... )
        False

        control_melody_interval: if True (default), then we check the interval between the
            melody and the next highest voice. (Otherwise we don't check this interval.)

        >>> validate_spacing([48, 60, 67, 88], SpacingConstraints())
        False
        >>> validate_spacing(
        ...     [48, 60, 67, 88], SpacingConstraints(control_melody_interval=False)
        ... )
        True

        avoid_melody_crossing: if True (default), then melody crossings are avoided even if
            control_melody_interval is False.

        >>> validate_spacing(
        ...     [48, 60, 67, 64], SpacingConstraints(control_melody_interval=False)
        ... )
        False
        >>> validate_spacing(
        ...     [48, 60, 67, 64],
        ...     SpacingConstraints(
        ...         control_melody_interval=False, avoid_melody_crossing=False
        ...     ),
        ... )
        True

        max_melody_interval: has no effect if control_melody_interval is False. Otherwise,
            specifies the maximum interval between the melody and the next highest voice.
            If None, we use max_adjacent_interval for this purpose.

        >>> validate_spacing(
        ...     [48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=8)
        ... )
        False
        >>> validate_spacing(
        ...     [48, 60, 67, 76],
        ...     SpacingConstraints(
        ...         max_adjacent_interval=9, control_melody_interval=False
        ...     ),
        ... )
        True
        >>> validate_spacing(
        ...     [48, 60, 67, 76],
        ...     SpacingConstraints(
        ...         max_adjacent_interval=9,
        ...         control_melody_interval=True,
        ...         max_melody_interval=12,
        ...     ),
        ... )
        True

        min_melody_interval: has no effect if control_melody_interval is False. Otherwise,
            specifies the minimum interval between the melody and the next highest voice.
            If avoid_melody_crossing is True, will be set to 0 if < 0.

        >>> validate_spacing(
        ...     [48, 60, 67, 70], SpacingConstraints(min_melody_interval=5)
        ... )
        False


    """

    max_adjacent_interval: int = 12
    min_adjacent_interval: int = 0
    min_total_interval: int = None  # type:ignore
    max_total_interval: int = None  # type:ignore
    control_bass_interval: bool = False
    max_bass_interval: int = None  # type:ignore
    min_bass_interval: int = None  # type:ignore
    control_melody_interval: bool = True
    max_melody_interval: int = None  # type:ignore
    min_melody_interval: int = None  # type:ignore

    avoid_bass_crossing: bool = True
    avoid_melody_crossing: bool = True

    def __post_init__(self):
        if self.min_total_interval is None:
            self.min_total_interval = -(2**30)
        if self.max_total_interval is None:
            self.max_total_interval = 2**31

        if self.max_bass_interval is None:
            self.max_bass_interval = (
                self.max_adjacent_interval if self.control_bass_interval else 2**31
            )

        if self.min_bass_interval is None:
            if self.control_bass_interval:
                if self.avoid_bass_crossing:
                    self.min_bass_interval = max(0, self.min_adjacent_interval)
                else:
                    self.min_bass_interval = self.min_adjacent_interval
            elif self.avoid_bass_crossing:
                self.min_bass_interval = 0
            else:
                self.min_bass_interval = -(2**30)

        if self.max_melody_interval is None:
            self.max_melody_interval = (
                self.max_adjacent_interval if self.control_melody_interval else 2**31
            )

        if self.min_melody_interval is None:
            if self.control_melody_interval:
                if self.avoid_melody_crossing:
                    self.min_melody_interval = max(0, self.min_adjacent_interval)
                else:
                    self.min_melody_interval = self.min_adjacent_interval
            elif self.avoid_melody_crossing:
                self.min_melody_interval = 0
            else:
                self.min_melody_interval = -(2**30)


def validate_spacing(
    spacing: t.Sequence[int],
    spacing_constraints: SpacingConstraints,
) -> bool:
    """
    # TODO: (Malcolm 2023-07-13) make this a method of SpacingConstraints?
    Args:

    >>> validate_spacing([48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=12))
    True
    >>> validate_spacing([48, 60, 67, 76], SpacingConstraints(max_adjacent_interval=9))
    True
    >>> validate_spacing(
    ...     [48, 60, 67, 76],
    ...     SpacingConstraints(max_adjacent_interval=9, control_bass_interval=True),
    ... )
    False
    >>> validate_spacing(
    ...     [48, 60, 67, 76],
    ...     SpacingConstraints(max_adjacent_interval=9, max_bass_interval=12),
    ... )
    True
    >>> validate_spacing(
    ...     [48, 60],
    ...     SpacingConstraints(
    ...         max_adjacent_interval=9,
    ...         control_bass_interval=True,
    ...         max_bass_interval=12,
    ...     ),
    ... )
    True
    >>> validate_spacing([48], SpacingConstraints(max_adjacent_interval=1))
    True
    >>> validate_spacing(
    ...     [48, 60, 67, 76],
    ...     SpacingConstraints(
    ...         max_adjacent_interval=9,
    ...         max_total_interval=24,
    ...         control_bass_interval=True,
    ...         max_bass_interval=12,
    ...     ),
    ... )
    False

    """
    if len(spacing) < 2:
        return True

    if spacing[-1] - spacing[0] > spacing_constraints.max_total_interval:
        return False

    if not (
        spacing_constraints.min_bass_interval
        <= spacing[1] - spacing[0]
        <= spacing_constraints.max_bass_interval
    ):
        return False

    for p1, p2 in zip(spacing[1:-2], spacing[2:-1]):
        if not (
            spacing_constraints.min_adjacent_interval
            <= p2 - p1
            <= spacing_constraints.max_adjacent_interval
        ):
            return False

    if len(spacing) > 2 and not (
        spacing_constraints.min_melody_interval
        <= spacing[-1] - spacing[-2]
        <= spacing_constraints.max_melody_interval
    ):
        return False

    return True


def _yield_spacing_helper(
    pitches_so_far: t.Tuple[Pitch, ...],
    remaining_pcs: t.List[PitchClass],
    range_constraints: RangeConstraints,
    spacing_constraints: SpacingConstraints,
    melody_pitch: Pitch | None,
    shuffled: bool = True,
):
    """Recursive sub-function used by yield_spacings()

    >>> list(
    ...     _yield_spacing_helper(
    ...         (),
    ...         [0, 4, 7, 0],
    ...         RangeConstraints(48, 84),
    ...         SpacingConstraints(max_adjacent_interval=9, control_bass_interval=True),
    ...         melody_pitch=None,
    ...         shuffled=True,
    ...     )
    ... )  # doctest: +SKIP
    [(48, 55, 64, 72), (48, 55, 60, 64), (48, 48, 52, 55), (48, 48, 55, 64),
     (48, 52, 60, 67), (48, 52, 55, 60)]

    >>> list(
    ...     _yield_spacing_helper(
    ...         (),
    ...         [0, 4, 7, 0],
    ...         RangeConstraints(48, 72),
    ...         SpacingConstraints(max_adjacent_interval=9, control_bass_interval=True),
    ...         melody_pitch=None,
    ...         shuffled=False,
    ...     )
    ... )  # doctest: +NORMALIZE_WHITESPACE
    [(48, 52, 55, 60), (48, 52, 60, 67), (48, 55, 64, 72), (48, 55, 60, 64),
     (48, 48, 52, 55), (48, 48, 55, 64)]

    Note: melody pitch, if provided, should be omitted from
    >>> list(
    ...     _yield_spacing_helper(
    ...         (),
    ...         [0, 7, 0],
    ...         RangeConstraints(48, 72, max_bass_pitch=48),
    ...         SpacingConstraints(max_adjacent_interval=9, control_bass_interval=True),
    ...         melody_pitch=76,
    ...         shuffled=False,
    ...     )
    ... )
    []

    >>> list(
    ...     _yield_spacing_helper(
    ...         (),
    ...         [0, 7, 0],
    ...         RangeConstraints(48, 84, max_bass_pitch=48),
    ...         SpacingConstraints(
    ...             max_adjacent_interval=9,
    ...             control_bass_interval=True,
    ...             control_melody_interval=False,
    ...         ),
    ...         melody_pitch=76,
    ...         shuffled=False,
    ...     )
    ... )
    [(48, 55, 60, 76), (48, 48, 55, 76)]
    """
    if not remaining_pcs:
        if melody_pitch:
            if not (
                spacing_constraints.max_melody_interval
                >= (melody_pitch - pitches_so_far[-1])
                >= spacing_constraints.min_melody_interval
            ):
                return

            yield pitches_so_far + (melody_pitch,)
        else:
            yield pitches_so_far
        return
    if not pitches_so_far:
        bass_pc = remaining_pcs.pop(0)

        if melody_pitch is None:
            min_bass_pitch = range_constraints.min_bass_pitch
        else:
            min_bass_pitch = max(
                range_constraints.min_bass_pitch,
                melody_pitch - spacing_constraints.max_total_interval,
            )

        for bass_pitch in get_all_in_range(
            bass_pc, min_bass_pitch, range_constraints.max_bass_pitch, shuffled=shuffled
        ):
            yield from _yield_spacing_helper(
                (bass_pitch,),
                remaining_pcs,
                range_constraints,
                spacing_constraints,
                melody_pitch,
                shuffled=shuffled,
            )
        return

    pc_range = range(len(remaining_pcs))
    if shuffled:
        pc_range = random.sample(pc_range, len(pc_range))

    past_pcs = set()

    for pc_i in pc_range:
        pc = remaining_pcs[pc_i]

        # We only want to do iterate through each pc once here, regardless of doublings
        if pc in past_pcs:
            continue
        past_pcs.add(pc)

        # Step 1: get adjacent interval bounds
        if len(pitches_so_far) == 1:
            max_adjacent_interval = spacing_constraints.max_bass_interval
            min_adjacent_interval = spacing_constraints.min_bass_interval
        elif melody_pitch is None and len(remaining_pcs) == 1:
            max_adjacent_interval = spacing_constraints.max_melody_interval
            min_adjacent_interval = spacing_constraints.min_melody_interval
        else:
            max_adjacent_interval = spacing_constraints.max_adjacent_interval
            min_adjacent_interval = spacing_constraints.min_adjacent_interval

        # Step 2 calculate min pitch
        min_pitch = max(
            pitches_so_far[-1] + min_adjacent_interval,
            range_constraints.min_pitch,
        )

        # Step 3 calculate max pitch
        max_pitch = min(
            range_constraints.max_pitch,
            pitches_so_far[0] + spacing_constraints.max_total_interval,
            pitches_so_far[-1] + max_adjacent_interval,
        )

        # Step 3 get remaining_pcs
        other_pcs = remaining_pcs[:pc_i] + remaining_pcs[pc_i + 1 :]

        for pitch in get_all_in_range(pc, min_pitch, max_pitch, shuffled=shuffled):
            yield from _yield_spacing_helper(
                pitches_so_far + (pitch,),
                other_pcs,
                range_constraints,
                spacing_constraints,
                melody_pitch,
                shuffled=shuffled,
            )


def yield_spacings(
    pcs: t.Sequence[PitchClass],
    range_constraints: RangeConstraints = RangeConstraints(),
    spacing_constraints: SpacingConstraints = SpacingConstraints(),
    bass_pitch: Pitch | None = None,
    melody_pitch: Pitch | None = None,
    shuffled: bool = False,
) -> t.Iterable[t.Tuple[Pitch]]:
    """
    This function doesn't double any pcs; the caller is responsible for that.

    >>> list(
    ...     yield_spacings((0, 0, 4, 7), bass_pitch=60)
    ... )  # doctest: +NORMALIZE_WHITESPACE
    [(60, 60, 64, 67), (60, 60, 67, 76), (60, 72, 76, 79), (60, 64, 72, 79),
     (60, 64, 67, 72), (60, 76, 79, 84), (60, 67, 72, 76), (60, 67, 76, 84)]
    >>> list(yield_spacings((0, 0, 4, 7), bass_pitch=60, melody_pitch=76))
    [(60, 60, 67, 76), (60, 67, 72, 76)]
    """
    pcs = list(pcs)

    if melody_pitch is not None:
        try:
            pcs.remove(melody_pitch % 12)
        except ValueError:
            raise ValueError("The pitch-class of `melody_pitch` must be in `pcs`")

    if bass_pitch:
        try:
            pcs.remove(bass_pitch % 12)
        except ValueError:
            raise ValueError("The pitch-class of `bass_pitch` must be in `pcs`")

        pitches_so_far = (bass_pitch,)
    else:
        pitches_so_far = ()

    yield from _yield_spacing_helper(
        pitches_so_far=pitches_so_far,
        remaining_pcs=pcs,
        range_constraints=range_constraints,
        spacing_constraints=spacing_constraints,
        melody_pitch=melody_pitch,
        shuffled=shuffled,
    )
    return
