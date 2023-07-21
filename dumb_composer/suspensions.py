import typing as t
from dataclasses import dataclass, field
from numbers import Number

from dumb_composer.constants import (
    DISSONANT_INTERVALS_ABOVE_BASS,
    DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES,
)
from dumb_composer.pitch_utils.chords import get_chords_from_rntxt  # used in doctests
from dumb_composer.pitch_utils.chords import Chord
from dumb_composer.pitch_utils.intervals import (
    reduce_compound_interval,
    smallest_pitch_class_interval,
)
from dumb_composer.pitch_utils.types import ChromaticInterval, Pitch, TimeStamp
from dumb_composer.time import Meter, MeterError


# TODO: (Malcolm 2023-07-20) update
def pitch_dissonant_against_chord(pitch: int, chord_pcs: t.Sequence[int]) -> bool:
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
    src_pitch: Pitch,
    suspension_chord: Chord,
    resolution_chord: Chord | None = None,
    suspension_chord_pcs_to_avoid: t.Container[Pitch] = frozenset(set()),
    resolution_chord_pcs_to_avoid: t.Container[Pitch] = frozenset(set()),
    resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
    resolve_up_by: t.Tuple[ChromaticInterval, ...] = (),
    enforce_dissonant: bool = False,
    suspension_must_belong_to_scale_of_suspension_chord: bool = True,
) -> list[Suspension]:
    """
    >>> rntxt = '''m1 C: IV b2 V b3 V43 b4 viio6
    ... m2 I  b2 IV64 b3 V65 b4 d: iiø42'''
    >>> IV, V, V43, viio6, I, IV64, V65, ii42 = get_chords_from_rntxt(rntxt)

    >>> find_suspensions(60, suspension_chord=V)
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)]

    We return a list because there can be more than one possible suspension.

    >>> for s in find_suspensions(71, suspension_chord=IV, resolve_up_by=(1,)):
    ...     print(s)
    ...
    Suspension(resolves_by=-2, dissonant=True, interval_above_bass=6)
    Suspension(resolves_by=1, dissonant=True, interval_above_bass=6)

    If the current pitch is already in the next chord, it can't be a suspension.

    >>> find_suspensions(67, suspension_chord=V43)
    []

    But the function isn't clever enough to recognize that the next chord can be
    interpreted as an incomplete V7 chord:

    >>> find_suspensions(67, suspension_chord=viio6)
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=5)]

    Determining whether a suspension is dissonant is tricky. Here are some
    special cases.

    >>> find_suspensions(69, suspension_chord=I)
    [Suspension(resolves_by=-2, dissonant=False, interval_above_bass=9)]

    >>> find_suspensions(67, suspension_chord=IV64)
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=7)]

    >>> find_suspensions(65, suspension_chord=ii42)
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=3)]

    We assume that the pitch to which the suspension resolves will not be
    sounding during the suspension *unless* the pitch is in the bass.

    >>> find_suspensions(62, IV64)[0].dissonant
    True

    By default suspensions have to belong to the scale of the suspension chord. We can
    turn off this requirement with the
    `suspension_must_belong_to_scale_of_suspension_chord` argument.
    >>> find_suspensions(
    ...     71, suspension_chord=ii42
    ... )  # No results because B-natural isn't in scale
    []
    >>> find_suspensions(
    ...     71,
    ...     suspension_chord=ii42,
    ...     suspension_must_belong_to_scale_of_suspension_chord=False,
    ... )
    [Suspension(resolves_by=-1, dissonant=False, interval_above_bass=9)]

    ------------------------------------------------------------------------------------
    Avoid tones
    ------------------------------------------------------------------------------------

    For various reasons (e.g., tendency tones already present in another voice) we may
    wish to avoid suspensions resolving to specific pitches. This can be obtained using
    the `suspension_chord_pcs_to_avoid` and `resolution_chord_pcs_to_avoid` arguments.


    >>> find_suspensions(72, suspension_chord=V)
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)]
    >>> find_suspensions(72, suspension_chord=V, suspension_chord_pcs_to_avoid={11})
    []
    >>> find_suspensions(72, suspension_chord=V, suspension_chord_pcs_to_avoid={10})
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)]
    >>> find_suspensions(72, suspension_chord=V65, suspension_chord_pcs_to_avoid={11})
    []

    >>> find_suspensions(
    ...     72,
    ...     suspension_chord=V,
    ...     resolution_chord=V,
    ...     resolution_chord_pcs_to_avoid={11},
    ... )
    []
    """
    if resolution_chord is None:
        resolution_chord = suspension_chord
        if resolution_chord_pcs_to_avoid:
            raise ValueError(
                "`resolution_chord_pcs_to_avoid` should be empty if `resolution_chord` is None"
            )
        resolution_chord_pcs_to_avoid = suspension_chord_pcs_to_avoid

    if (
        suspension_must_belong_to_scale_of_suspension_chord
        and src_pitch % 12 not in suspension_chord.scale_pcs
    ):
        return []

    if src_pitch % 12 in resolution_chord.pcs:
        return []

    out = []
    for resolve_by in (resolve_down_by, resolve_up_by):
        for resolution_interval in resolve_by:
            resolution_pc = (src_pitch + resolution_interval) % 12
            if (
                resolution_pc in resolution_chord_pcs_to_avoid
                or resolution_pc not in resolution_chord.pcs
            ):
                continue

            comparison_pc = None
            if resolution_chord == suspension_chord:
                comparison_pc = resolution_pc
            else:
                to_continue = True
                for interval in resolve_by:
                    comparison_pc = (src_pitch + interval) % 12
                    if (
                        comparison_pc not in suspension_chord_pcs_to_avoid
                        and comparison_pc in suspension_chord.pcs
                    ):
                        to_continue = False
                        break
                if to_continue:
                    continue
            assert comparison_pc is not None

            interval_above_bass = reduce_compound_interval(
                src_pitch - suspension_chord.foot
            )

            other_pcs = list(suspension_chord.pcs)
            if comparison_pc != suspension_chord.foot:
                other_pcs.remove(comparison_pc)
            dissonant = pitch_dissonant_against_chord(src_pitch, other_pcs)
            if enforce_dissonant and not dissonant:
                continue

            out.append(
                Suspension(
                    resolves_by=resolution_interval,
                    dissonant=dissonant,
                    interval_above_bass=interval_above_bass,
                )
            )

    return out


# def find_suspensions_old(
#     src_pitch: Pitch,
#     dst_chord_pcs: t.Sequence[PitchClass],
#     dst_scale_pcs: t.Optional[t.Sequence[PitchClass]] = None,
#     resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
#     resolve_up_by: t.Tuple[ChromaticInterval, ...] = (1,),
# ) -> t.List[Suspension]:
#     """
#     >>> find_suspensions(60, (7, 11, 2))
#     [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=5)]

#     We return a list because there can be more than one possible suspension.

#     >>> for s in find_suspensions(71, (5, 9, 0)):
#     ...     print(s)
#     ...
#     Suspension(resolves_by=-2, dissonant=True, interval_above_bass=6)
#     Suspension(resolves_by=1, dissonant=True, interval_above_bass=6)

#     If the current pitch is already in the next chord, it can't be a suspension.

#     >>> find_suspensions(67, (2, 5, 7, 11))
#     []

#     But the function isn't clever enough to recognize that the next chord can be
#     interpreted as an incomplete V7 chord:

#     >>> find_suspensions(67, (2, 5, 11))
#     [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=5)]

#     Determining whether a suspension is dissonant is tricky. Here are some
#     special cases.

#     >>> find_suspensions(69, (0, 4, 7))
#     [Suspension(resolves_by=-2, dissonant=False, interval_above_bass=9)]

#     >>> find_suspensions(67, (0, 5, 9))
#     [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=7)]

#     >>> find_suspensions(65, (2, 4, 7, 10))
#     [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=3)]

#     We assume that the pitch to which the suspension resolves will not be
#     sounding during the suspension *unless* the pitch is in the bass.

#     >>> find_suspensions(62, (0, 5, 9))[0].dissonant
#     True

#     We can disable "chromatic" suspensions (i.e., suspensions that belong to the
#     chord/scale of the preparation but not to the chord/scale of the suspension)
#     by providing the `scale_pcs` argument; if this argument is not None, the
#     `src_pitch` has to belong to the scale.
#     >>> find_suspensions(69, (0, 4, 7), (0, 2, 3, 5, 7, 8, 11))
#     []
#     """

#     # TODO: (Malcolm 2023-07-20) update so takes intermediate_chord_pcs
#     def _append(src_pitch, interval, dst_pc):
#         interval_above_bass = (src_pitch - dst_chord_pcs[0]) % 12
#         other_pcs = list(dst_chord_pcs)
#         if dst_pc != other_pcs[0]:
#             other_pcs.remove(dst_pc)
#         dissonant = pitch_dissonant_against_chord(src_pitch, other_pcs)
#         # TODO: (Malcolm 2023-07-18) do we want to update expected_resolution_interval
#         #   to take account of the fact that the bass may change?
#         expected_resolution_interval = (interval_above_bass + interval) % 12
#         out.append(
#             Suspension(
#                 interval,
#                 dissonant,
#                 interval_above_bass,
#                 score=TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS[
#                     expected_resolution_interval
#                 ],
#             )
#         )

#     out = []
#     if dst_scale_pcs is not None and src_pitch % 12 not in dst_scale_pcs:
#         return out
#     dst_chord_set = set(dst_chord_pcs)
#     if src_pitch % 12 in dst_chord_set:
#         return out
#     for interval in resolve_down_by + resolve_up_by:
#         if (dst_pc := (src_pitch + interval) % 12) in dst_chord_set:
#             _append(src_pitch, interval, dst_pc)
#     return out


def find_bass_suspension(
    src_pitch: Pitch,
    suspension_chord: Chord,
    resolution_chord: Chord | None = None,
    resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
    resolve_up_by: t.Tuple[ChromaticInterval, ...] = (),
    enforce_dissonant: bool = True,
    suspension_must_belong_to_scale_of_suspension_chord: bool = True,
) -> list[Suspension]:
    """
    The returned list contains at most 1 element but we return a list for a consistent
    API with find_suspensions.

    The following returns no suspensions because `src_pitch` is not dissonant against
    `dst_chord_pcs`
    >>> rntxt = '''m1 F: ii6 b2 V6/V b3 V42
    ... m2 I b2 vi6 b3 B: I6'''
    >>> ii6, V6_of_V, V42, I, vi6, B_major = get_chords_from_rntxt(rntxt)
    >>> find_bass_suspension(src_pitch=48, suspension_chord=V42)
    []

    If we add an intermediate chord, however, a suspension is possible:
    >>> find_bass_suspension(src_pitch=48, suspension_chord=ii6, resolution_chord=V42)
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)]

    We can also obtain a suspension even when the pitch of resolution is not in the
    intermediate chord, provided a valid suspension resolution in the same direction
    *is* in that chord:
    >>> find_bass_suspension(
    ...     src_pitch=48, suspension_chord=V6_of_V, resolution_chord=V42
    ... )
    [Suspension(resolves_by=-2, dissonant=True, interval_above_bass=0)]
    >>> find_bass_suspension(
    ...     src_pitch=48, suspension_chord=ii6, resolution_chord=V6_of_V
    ... )
    [Suspension(resolves_by=-1, dissonant=True, interval_above_bass=0)]

    However, if there is no pitch of resolution in the same direction in the
    intermediate chord, there is no suspension:
    >>> find_bass_suspension(src_pitch=48, suspension_chord=I, resolution_chord=ii6)
    []
    >>> find_bass_suspension(src_pitch=52, suspension_chord=vi6, resolve_up_by=(1,))
    [Suspension(resolves_by=1, dissonant=True, interval_above_bass=0)]
    >>> find_bass_suspension(
    ...     src_pitch=52,
    ...     suspension_chord=B_major,
    ...     resolution_chord=vi6,
    ...     resolve_up_by=(1,),
    ... )
    []
    """
    if resolution_chord is None:
        resolution_chord = suspension_chord

    if (
        suspension_must_belong_to_scale_of_suspension_chord
        and src_pitch % 12 not in suspension_chord.scale_pcs
    ):
        return []

    if src_pitch % 12 == resolution_chord.foot:
        return []

    resolution_interval = smallest_pitch_class_interval(
        src_pitch, resolution_chord.foot
    )

    if resolution_interval in resolve_down_by:
        resolve_by = resolve_down_by
    elif resolution_interval in resolve_up_by:
        resolve_by = resolve_up_by
    else:
        return []

    dissonant = pitch_dissonant_against_chord(
        pitch=src_pitch, chord_pcs=suspension_chord.non_foot_pcs
    )
    if enforce_dissonant and not dissonant:
        return []
    if resolution_chord == suspension_chord:
        return [Suspension(resolution_interval, dissonant, interval_above_bass=0)]

    intermediate_interval = smallest_pitch_class_interval(
        src_pitch, suspension_chord.foot
    )
    if intermediate_interval in resolve_by:
        return [Suspension(resolution_interval, dissonant, interval_above_bass=0)]

    return []


def find_suspension_release_times(
    start: TimeStamp,
    stop: TimeStamp,
    meter: Meter,
    max_weight_diff: t.Optional[int] = None,
    max_suspension_dur: t.Union[str, Number] = "bar",
    include_stop: bool = False,
) -> t.List[TimeStamp]:
    """Returns a list of times at which suspension releases could occur.

    By default, the times are exclusive of `stop`:
    >>> find_suspension_release_times(0.0, 4.0, Meter("4/4"), max_weight_diff=2)
    [Fraction(2, 1), Fraction(1, 1)]

    However, this behavior can be changed with the `include_stop` argument:
    >>> find_suspension_release_times(
    ...     0.0, 4.0, Meter("4/4"), max_weight_diff=2, include_stop=True
    ... )
    [Fraction(4, 1), Fraction(2, 1), Fraction(1, 1)]

    # TODO behavior is inconsistent between 9/8 and 3/4. FIX!
    >>> find_suspension_release_times(0.0, 4.5, Meter("9/8"), max_weight_diff=2)
    [Fraction(3, 1), Fraction(1, 1)]

    By default, suspensions can be at most one bar long:

    >>> find_suspension_release_times(0.0, 16.0, Meter("4/4"), max_weight_diff=2)
    [Fraction(4, 1), Fraction(2, 1), Fraction(1, 1)]
    >>> find_suspension_release_times(0.0, 12.0, Meter("3/4"), max_weight_diff=1)
    [Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)]

    >>> find_suspension_release_times(16.0, 32.0, Meter("4/4"), max_weight_diff=2)
    [Fraction(20, 1), Fraction(18, 1), Fraction(17, 1)]

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
                start,
                stop,
                include_start=False,
                return_first=False,
                include_stop=include_stop,
            )
        except MeterError:
            break
        # print(res_onset, res_weight)
        if max_weight_diff is not None and diss_weight - res_weight > max_weight_diff:
            break
        if (
            diss_weight >= res_weight
            and res_onset - diss_onset <= max_suspension_dur  # type:ignore
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
