from dataclasses import dataclass, field
from numbers import Number
import typing as t

from dumb_composer.constants import (
    DISSONANT_INTERVALS_ABOVE_BASS,
    DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES,
)
from .time import Meter, MeterError


def pitch_dissonant_against_chord(
    pitch: int, chord_pcs: t.Sequence[int]
) -> bool:
    """The first item of chord_pcs is understood to be the bass.

    This function should eventually be replaced by a function that
    understands diatonic intervals (so it can recognize that, e.g., a diminished
    fourth is dissonant).

    >>> pitch_dissonant_against_chord(67, (0, 4, 7))
    False
    >>> pitch_dissonant_against_chord(69, (0, 4, 7))
    True
    >>> pitch_dissonant_against_chord(69, (0, 4))
    False
    >>> pitch_dissonant_against_chord(65, (0, 8))
    True

    Testing special cases:

    >>> pitch_dissonant_against_chord(65, (0,))
    True
    >>> pitch_dissonant_against_chord(67, (0,))
    False
    >>> pitch_dissonant_against_chord(67, ())
    False
    """
    if not chord_pcs:
        return False
    if (pitch - chord_pcs[0]) % 12 in DISSONANT_INTERVALS_ABOVE_BASS:
        return True
    if any(
        (pitch - pc) % 12 in DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES
        for pc in chord_pcs[1:]
    ):
        return True
    return False


@dataclass
class Suspension:
    resolves_by: int
    dissonant: bool
    interval_above_bass: int
    # "score" is meant to be used to weight how likely we are to use
    #   each suspension.
    score: float = field(default=1.0, repr=False)


def find_suspensions(
    current_pitch: int,
    next_chord_pcs: t.Sequence[int],
    next_scale_pcs: t.Optional[t.Sequence[int]] = None,
    resolve_down_by: t.Tuple = (-1, -2),
    resolve_up_by: t.Tuple = (1,),
) -> t.List[Suspension]:
    """
    >>> find_suspensions(60, (7, 11, 2))
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)]

    We return a list because there can be more than one possible suspension.

    >>> for s in find_suspensions(71, (5, 9, 0)):
    ...     print(s)
    Suspension(resolves_by=-2, dissonant=True, interval_above_bass=6)
    Suspension(resolves_by=1, dissonant=True, interval_above_bass=6)

    If the current pitch is already in the next chord, it can't be a suspension.

    >>> find_suspensions(67, (2, 5, 7, 11))
    []

    But the function isn't clever enough to recognize that this chord can be
    interpreted as an incomplete V7 chord:

    >>> find_suspensions(67, (2, 5, 11))
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=5)]

    Determining whether a suspension is dissonant is tricky. Here are some
    special cases.

    >>> find_suspensions(69, (0, 4, 7))
    [Suspension(resolves_by=-2, dissonant=False, interval_above_bass=9)]

    >>> find_suspensions(67, (0, 5, 9))
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=7)]

    >>> find_suspensions(65, (2, 4, 7, 10))
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=3)]

    We assume that the pitch to which the suspension resolves will not be
    sounding during the suspension *unless* the pitch is in the bass.

    >>> find_suspensions(62, (0, 5, 9))[0].dissonant
    True

    We can disable "chromatic" suspensions (i.e., suspensions that belong to the
    chord/scale of the preparation but not to the chord/scale of the suspension)
    by providing the `scale_pcs` argument; if this argument is not None, the
    current_pitch has to belong to the scale.
    >>> find_suspensions(69, (0, 4, 7), (0, 2, 3, 5, 7, 8, 11))
    []
    """

    def _append(current_pitch, interval, next_pc):
        interval_above_bass = (current_pitch - next_chord_pcs[0]) % 12
        other_pcs = list(next_chord_pcs)
        if next_pc != other_pcs[0]:
            other_pcs.remove(next_pc)
        dissonant = pitch_dissonant_against_chord(current_pitch, other_pcs)
        out.append(Suspension(interval, dissonant, interval_above_bass))

    out = []
    if next_scale_pcs is not None and current_pitch % 12 not in next_scale_pcs:
        return out
    next_chord_set = set(next_chord_pcs)
    if current_pitch % 12 in next_chord_set:
        return out
    for interval in resolve_down_by + resolve_up_by:
        if (next_pc := (current_pitch + interval) % 12) in next_chord_set:
            _append(current_pitch, interval, next_pc)
    return out


def find_suspension_releases(
    start: Number,
    stop: Number,
    meter: Meter,
    max_weight_diff: t.Optional[int] = None,
    max_suspension_dur: t.Union[str, Number] = "bar",
) -> t.List[Number]:
    """
    # TODO behavior is inconsistent between 9/8 and 3/4. FIX!
    >>> find_suspension_releases(
    ...     0.0, 4.5, Meter("9/8"), max_weight_diff=2)
    [Fraction(3, 1), Fraction(1, 1)]

    By default, suspensions can be at most one bar long:

    >>> find_suspension_releases(
    ...     0.0, 16.0, Meter("4/4"), max_weight_diff=2)
    [Fraction(4, 1), Fraction(2, 1), Fraction(1, 1)]
    >>> find_suspension_releases(
    ...     0.0, 12.0, Meter("3/4"), max_weight_diff=1)
    [Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)]
    """
    out = []
    if max_suspension_dur == "bar":
        max_suspension_dur = meter.bar_dur
    diss_onset, diss_weight = start, meter.weight(start)
    # print(diss_onset, diss_weight)
    while True:
        # print(start, stop)
        try:
            res_onset, res_weight = meter.get_onset_of_greatest_weight_between(
                start, stop, include_start=False, return_first=False
            )
        except MeterError:
            break
        # print(res_onset, res_weight)
        if (
            max_weight_diff is not None
            and diss_weight - res_weight > max_weight_diff
        ):
            break
        if (
            diss_weight >= res_weight
            and res_onset - diss_onset <= max_suspension_dur
        ):
            out.append(res_onset)
        if res_weight == meter.min_weight:
            break
        # print(stop)
        stop, _ = meter.get_onset_of_greatest_weight_between(
            start, stop, include_start=False, return_first=meter.is_compound
        )
    # TODO suspensions releases should have a "score" that indicates
    #   how likely they are to be employed.
    return out
