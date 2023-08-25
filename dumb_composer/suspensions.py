import typing as t
from itertools import chain, combinations
from numbers import Number

from dumb_composer.chords.chords import get_chords_from_rntxt  # used in doctests
from dumb_composer.chords.chords import Chord, Tendency
from dumb_composer.constants import (
    DISSONANT_INTERVALS_ABOVE_BASS,
    DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES,
    TWELVE_TET_SUSPENSION_RESOLUTION_INTERVAL_WEIGHTS,
)
from dumb_composer.pitch_utils.aliases import Third
from dumb_composer.pitch_utils.intervals import (
    reduce_compound_interval,
    smallest_pitch_class_interval,
)
from dumb_composer.pitch_utils.types import (
    ChordFactor,
    ChromaticInterval,
    Pitch,
    ScalarInterval,
    Suspension,
    TimeStamp,
    Voice,
)
from dumb_composer.time import Meter, MeterError


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


def validate_intervals_among_suspensions(
    suspension_pitches: t.Iterable[Pitch],
    bass_suspension_pitch: Pitch | None = None,
    whitelisted_intervals: t.Container[ChromaticInterval] = frozenset({3, 4, 8, 9}),
    secondary_whitelisted_intervals: t.Container[ChromaticInterval] = frozenset({5, 6}),
) -> bool:
    """
    Validates the intervals between simultaneous suspensions.

    The default settings are as follows:
        - imperfect consonances are always OK
        - perfect/augmented 4ths are OK *if* there is also at least one imperfect
          consonance

    Note: since we're only considering chromatic intervals some enharmonic equivalents
    will sneak through (e.g., diminished 7ths, diminished 4ths).

    >>> validate_intervals_among_suspensions((60,), bass_suspension_pitch=48)
    False
    >>> validate_intervals_among_suspensions((48, 60))
    False
    >>> validate_intervals_among_suspensions((55, 60))
    False
    >>> validate_intervals_among_suspensions((55, 60, 64))
    True
    >>> validate_intervals_among_suspensions((60, 64), bass_suspension_pitch=48)
    False
    >>> validate_intervals_among_suspensions((55, 61, 64))
    True

    ------------------------------------------------------------------------------------
    Special cases
    ------------------------------------------------------------------------------------

    >>> validate_intervals_among_suspensions((60,))
    True
    >>> validate_intervals_among_suspensions((), bass_suspension_pitch=48)
    True
    >>> validate_intervals_among_suspensions(())
    True

    """
    if bass_suspension_pitch is not None:
        for pitch in suspension_pitches:
            interval = (pitch - bass_suspension_pitch) % 12
            if interval not in whitelisted_intervals:
                return False

    has_whitelisted_interval = False
    has_secondary_interval = False
    for pair_of_pitches in combinations(suspension_pitches, r=2):
        low_pitch, high_pitch = sorted(pair_of_pitches)
        interval = (high_pitch - low_pitch) % 12
        if interval in whitelisted_intervals:
            has_whitelisted_interval = True
        elif interval in secondary_whitelisted_intervals:
            has_secondary_interval = True
        else:
            return False

    if has_secondary_interval and not has_whitelisted_interval:
        return False

    return True


def find_suspensions(
    src_pitch: Pitch,
    preparation_chord: Chord,
    suspension_chord: Chord,
    resolution_chord: Chord | None = None,
    suspension_chord_pcs_to_avoid: t.Container[Pitch] = frozenset(set()),
    resolution_chord_pcs_to_avoid: t.Container[Pitch] = frozenset(set()),
    resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
    resolve_up_by: t.Tuple[ChromaticInterval, ...] = (),
    enforce_dissonant: bool = False,
    suspension_must_belong_to_scale_of_suspension_chord: bool = True,
    other_suspended_nonbass_pitches: tuple[Pitch, ...] = (),
    other_suspended_bass_pitch: Pitch | None = None,
    contrary_tendency_score_factor: float = 0.25,
    similar_tendency_score_factor: float = 1.25,
    forbidden_suspension_intervals_above_bass: tuple[ChromaticInterval, ...] = (),
) -> list[Suspension]:
    """
    >>> rntxt = '''m1 C: IV b2 V b3 V43 b4 viio6
    ... m2 I  b2 IV64 b3 V65 b4 d: iiø42'''
    >>> IV, V, V43, viio6, I, IV64, V65, ii42 = get_chords_from_rntxt(rntxt)

    >>> find_suspensions(60, preparation_chord=I, suspension_chord=V)
    [Suspension(pitch=60, resolves_by=-1, dissonant=True, interval_above_bass=5, ...

    We return a list because there can be more than one possible suspension.

    >>> for s in find_suspensions(
    ...     71, preparation_chord=V, suspension_chord=IV, resolve_up_by=(1,)
    ... ):
    ...     print(s)
    Suspension(pitch=71, resolves_by=-2, dissonant=True, interval_above_bass=6, ...
    Suspension(pitch=71, resolves_by=1, dissonant=True, interval_above_bass=6, ...

    If the current pitch is already in the next chord, it can't be a suspension.

    >>> find_suspensions(67, preparation_chord=I, suspension_chord=V43)
    []

    But the function isn't clever enough to recognize that the next chord can be
    interpreted as an incomplete V7 chord:

    >>> find_suspensions(67, preparation_chord=I, suspension_chord=viio6)
    [Suspension(pitch=67, resolves_by=-2, dissonant=True, interval_above_bass=5, ...

    Determining whether a suspension is dissonant is tricky. Here are some
    special cases.

    >>> find_suspensions(69, preparation_chord=IV, suspension_chord=I)
    [Suspension(pitch=69, resolves_by=-2, dissonant=False, interval_above_bass=9, ...

    >>> find_suspensions(67, preparation_chord=I, suspension_chord=IV64)
    [Suspension(pitch=67, resolves_by=-2, dissonant=True, interval_above_bass=7, ...

    >>> find_suspensions(65, preparation_chord=IV, suspension_chord=ii42)
    [Suspension(pitch=65, resolves_by=-1, dissonant=True, interval_above_bass=3, ...

    We assume that the pitch to which the suspension resolves will not be
    sounding during the suspension *unless* the pitch is in the bass.

    >>> find_suspensions(62, preparation_chord=V, suspension_chord=IV64)[0].dissonant
    True

    By default suspensions have to belong to the scale of the suspension chord. We can
    turn off this requirement with the
    `suspension_must_belong_to_scale_of_suspension_chord` argument.
    >>> find_suspensions(
    ...     71, preparation_chord=V, suspension_chord=ii42
    ... )  # No results because B-natural isn't in scale
    []
    >>> find_suspensions(
    ...     71,
    ...     preparation_chord=V,
    ...     suspension_chord=ii42,
    ...     suspension_must_belong_to_scale_of_suspension_chord=False,
    ... )
    [Suspension(pitch=71, resolves_by=-1, dissonant=False, interval_above_bass=9, ...

    ------------------------------------------------------------------------------------
    Avoid tones
    ------------------------------------------------------------------------------------

    For various reasons (e.g., tendency tones already present in another voice) we may
    wish to avoid suspensions resolving to specific pitches. This can be obtained using
    the `suspension_chord_pcs_to_avoid` and `resolution_chord_pcs_to_avoid` arguments.


    >>> find_suspensions(72, preparation_chord=I, suspension_chord=V)
    [Suspension(pitch=72, resolves_by=-1, dissonant=True, interval_above_bass=5, ...
    >>> find_suspensions(
    ...     72,
    ...     preparation_chord=I,
    ...     suspension_chord=V,
    ...     suspension_chord_pcs_to_avoid={11},
    ... )
    []
    >>> find_suspensions(
    ...     72,
    ...     preparation_chord=I,
    ...     suspension_chord=V,
    ...     suspension_chord_pcs_to_avoid={10},
    ... )
    [Suspension(pitch=72, resolves_by=-1, dissonant=True, interval_above_bass=5, ...
    >>> find_suspensions(
    ...     72,
    ...     preparation_chord=I,
    ...     suspension_chord=V65,
    ...     suspension_chord_pcs_to_avoid={11},
    ... )
    []

    >>> find_suspensions(
    ...     72,
    ...     preparation_chord=I,
    ...     suspension_chord=V,
    ...     resolution_chord=V,
    ...     resolution_chord_pcs_to_avoid={11},
    ... )
    []

    ------------------------------------------------------------------------------------
    `contrary_tendency_score_factor` and `similar_tendency_score_factor`
    ------------------------------------------------------------------------------------

    We probably don't want so many suspensions that contradict tendency tones, and maybe
    we want more suspensions that proceed in the same direction as the tendency.
    Therefore we scale the suspension score by these arguments depending on the tendency
    of the pitch.

    >>> rntxt = '''m1 Ab: I b2 Db: V b3 I6 b4 vi6'''
    >>> Ab, V_of_Db, Db6, bb6 = get_chords_from_rntxt(rntxt)

    No tendency tone, score is 5.0:
    >>> find_suspensions(
    ...     60, preparation_chord=Ab, suspension_chord=Db6, resolve_up_by=(1,)
    ... )
    [Suspension(pitch=60, resolves_by=1, dissonant=False, interval_above_bass=7, ...

    Suspension resolves similar to tendency tone, score is increased to 6.25:
    >>> find_suspensions(
    ...     60, preparation_chord=V_of_Db, suspension_chord=Db6, resolve_up_by=(1,)
    ... )
    [Suspension(pitch=60, resolves_by=1, dissonant=False, interval_above_bass=7, ...

    Suspension resolves contrary to tendency tone, score is reduced to 1.25:
    >>> find_suspensions(60, preparation_chord=V_of_Db, suspension_chord=bb6)
    [Suspension(pitch=60, resolves_by=-2, dissonant=True, interval_above_bass=11, ...
    """
    if resolution_chord is None:
        resolution_chord = suspension_chord
        if resolution_chord_pcs_to_avoid:
            raise ValueError(
                "`resolution_chord_pcs_to_avoid` should be empty if `resolution_chord` is None"
            )
        resolution_chord_pcs_to_avoid = suspension_chord_pcs_to_avoid

    if not validate_intervals_among_suspensions(
        suspension_pitches=chain((src_pitch,), other_suspended_nonbass_pitches),
        bass_suspension_pitch=other_suspended_bass_pitch,
    ):
        return []

    if (
        suspension_must_belong_to_scale_of_suspension_chord
        and src_pitch % 12 not in suspension_chord.scale_pcs
    ):
        return []

    if src_pitch % 12 in resolution_chord.pcs:
        return []

    src_pitch_tendency = preparation_chord.get_pitch_tendency(src_pitch)

    out = []
    interval_above_bass = reduce_compound_interval(src_pitch - suspension_chord.foot)
    if interval_above_bass in forbidden_suspension_intervals_above_bass:
        return out

    for tendency, resolve_by in (
        (Tendency.DOWN, resolve_down_by),
        (Tendency.UP, resolve_up_by),
    ):
        for resolution_interval in resolve_by:
            resolution_pc = (src_pitch + resolution_interval) % 12
            if (
                resolution_pc in resolution_chord_pcs_to_avoid
                or resolution_pc not in resolution_chord.pcs
            ):
                continue

            displaced_pc = None
            if resolution_chord == suspension_chord:
                displaced_pc = resolution_pc
            else:
                to_continue = True
                for interval in resolve_by:
                    displaced_pc = (src_pitch + interval) % 12
                    if (
                        displaced_pc not in suspension_chord_pcs_to_avoid
                        and displaced_pc in suspension_chord.pcs
                    ):
                        to_continue = False
                        break
                if to_continue:
                    continue
            assert displaced_pc is not None

            other_pcs = list(suspension_chord.pcs)
            if displaced_pc != suspension_chord.foot:
                other_pcs.remove(displaced_pc)
            dissonant = pitch_dissonant_against_chord(src_pitch, other_pcs)
            if enforce_dissonant and not dissonant:
                continue

            # expected_resolution_interval = reduce_compound_interval(
            #     (src_pitch + resolution_interval) - resolution_chord.foot
            # )

            if src_pitch_tendency is Tendency.NONE:
                score_factor = 1.0
            elif src_pitch_tendency == tendency:
                score_factor = similar_tendency_score_factor
            else:
                score_factor = contrary_tendency_score_factor

            displaced_chord_factor = suspension_chord.pitch_to_chord_factor(
                displaced_pc
            )
            chord_factor_weight = suspension_chord.suspension_weight_per_chord_factor[
                displaced_chord_factor
            ]

            score = (
                # TODO: (Malcolm 2023-07-23) restore
                # TWELVE_TET_SUSPENSION_RESOLUTION_INTERVAL_WEIGHTS[
                #     expected_resolution_interval
                # ] *
                score_factor
                * chord_factor_weight
            )

            out.append(
                Suspension(
                    pitch=src_pitch,
                    resolves_by=resolution_interval,
                    dissonant=dissonant,
                    interval_above_bass=interval_above_bass,
                    score=score,
                )
            )

    return out


def find_bass_suspension(
    src_pitch: Pitch,
    preparation_chord: Chord,
    suspension_chord: Chord,
    resolution_chord: Chord | None = None,
    resolve_down_by: t.Tuple[ChromaticInterval, ...] = (-1, -2),
    resolve_up_by: t.Tuple[ChromaticInterval, ...] = (),
    enforce_dissonant: bool = True,
    suspension_must_belong_to_scale_of_suspension_chord: bool = True,
    other_suspended_pitches: tuple[Pitch, ...] = (),
    contrary_tendency_score_factor: float = 0.25,
    similar_tendency_score_factor: float = 1.25,
) -> list[Suspension]:
    """
    The returned list contains at most 1 element but we return a list for a consistent
    API with find_suspensions.

    The following returns no suspensions because `src_pitch` is not dissonant against
    `dst_chord_pcs`
    >>> rntxt = '''m1 F: ii6 b2 V6/V b3 V b4 V42
    ... m2 I b2 V6 b3 vi6 b4 B: I6'''
    >>> ii6, V6_of_V, V, V42, I, V6, vi6, B_major = get_chords_from_rntxt(rntxt)
    >>> find_bass_suspension(src_pitch=48, preparation_chord=V, suspension_chord=V42)
    []

    If we add an intermediate chord, however, a suspension is possible:
    >>> find_bass_suspension(
    ...     src_pitch=48,
    ...     preparation_chord=V,
    ...     suspension_chord=ii6,
    ...     resolution_chord=V42,
    ... )
    [Suspension(pitch=48, resolves_by=-2, dissonant=True, interval_above_bass=0, ...

    We can also obtain a suspension even when the pitch of resolution is not in the
    intermediate chord, provided a valid suspension resolution in the same direction
    *is* in that chord:
    >>> find_bass_suspension(
    ...     src_pitch=48,
    ...     preparation_chord=V,
    ...     suspension_chord=V6_of_V,
    ...     resolution_chord=V42,
    ... )
    [Suspension(pitch=48, resolves_by=-2, dissonant=True, interval_above_bass=0, ...
    >>> find_bass_suspension(
    ...     src_pitch=48,
    ...     preparation_chord=V,
    ...     suspension_chord=ii6,
    ...     resolution_chord=V6_of_V,
    ... )
    [Suspension(pitch=48, resolves_by=-1, dissonant=True, interval_above_bass=0, ...

    However, if there is no pitch of resolution in the same direction in the
    intermediate chord, there is no suspension:
    >>> find_bass_suspension(
    ...     src_pitch=48, preparation_chord=V, suspension_chord=I, resolution_chord=ii6
    ... )
    []
    >>> find_bass_suspension(
    ...     src_pitch=52, preparation_chord=V6, suspension_chord=vi6, resolve_up_by=(1,)
    ... )
    [Suspension(pitch=52, resolves_by=1, dissonant=True, interval_above_bass=0, ...
    >>> find_bass_suspension(
    ...     src_pitch=52,
    ...     preparation_chord=V6,
    ...     suspension_chord=B_major,
    ...     resolution_chord=vi6,
    ...     resolve_up_by=(1,),
    ... )
    []
    """
    if resolution_chord is None:
        resolution_chord = suspension_chord

    if not validate_intervals_among_suspensions(
        suspension_pitches=other_suspended_pitches, bass_suspension_pitch=src_pitch
    ):
        return []

    if (
        suspension_must_belong_to_scale_of_suspension_chord
        and src_pitch % 12 not in suspension_chord.scale_pcs
    ):
        return []

    if src_pitch % 12 == resolution_chord.foot:
        return []

    src_pitch_tendency = preparation_chord.get_pitch_tendency(src_pitch)

    resolution_interval = smallest_pitch_class_interval(
        src_pitch, resolution_chord.foot
    )

    if resolution_interval in resolve_down_by:
        resolve_by = resolve_down_by
        tendency = Tendency.DOWN
    elif resolution_interval in resolve_up_by:
        resolve_by = resolve_up_by
        tendency = Tendency.UP
    else:
        return []

    dissonant = pitch_dissonant_against_chord(
        pitch=src_pitch, chord_pcs=suspension_chord.non_foot_pcs
    )
    if enforce_dissonant and not dissonant:
        return []

    if src_pitch_tendency is Tendency.NONE:
        score_factor = 1.0
    elif src_pitch_tendency == tendency:
        score_factor = similar_tendency_score_factor
    else:
        score_factor = contrary_tendency_score_factor

    expected_resolution_interval = reduce_compound_interval(
        (src_pitch + resolution_interval) - resolution_chord.foot
    )

    displaced_chord_factor = suspension_chord.pitch_to_chord_factor(
        suspension_chord.foot
    )
    chord_factor_weight = suspension_chord.suspension_weight_per_chord_factor[
        displaced_chord_factor
    ]

    score = (
        # TODO: (Malcolm 2023-07-23) restore?
        # TWELVE_TET_SUSPENSION_RESOLUTION_INTERVAL_WEIGHTS[expected_resolution_interval]
        chord_factor_weight
        * score_factor
    )

    if resolution_chord == suspension_chord:
        return [
            Suspension(
                src_pitch,
                resolution_interval,
                dissonant,
                interval_above_bass=0,
                score=score,
            )
        ]

    intermediate_interval = smallest_pitch_class_interval(
        src_pitch, suspension_chord.foot
    )
    if intermediate_interval in resolve_by:
        return [
            Suspension(
                src_pitch,
                resolution_interval,
                dissonant,
                interval_above_bass=0,
                score=score,
            )
        ]

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

    The returned list is in reverse sorted order (i.e., the latest release is first,
    the earliest release is last).

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

    while True:
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

        if max_weight_diff is not None and diss_weight - res_weight > max_weight_diff:
            break
        if (
            diss_weight >= res_weight
            and res_onset - diss_onset <= max_suspension_dur  # type:ignore
        ):
            out.append(res_onset)
        if res_weight == meter.min_weight:
            break
        try:
            stop, _ = meter.get_onset_of_greatest_weight_between(
                start, stop, include_start=False, return_first=meter.is_compound
            )
        except MeterError:
            break

    # TODO suspensions releases should have a "score" that indicates
    #   how likely they are to be employed.
    return out


def validate_suspension_resolution(
    resolution_pitch: Pitch,
    other_pitches: t.Sequence[Pitch],
    resolution_chord: Chord,
    prev_melody_pitch: Pitch | None = None,
    melody_pitch: Pitch | None = None,
) -> bool:
    # 1: if resolution pc is a tendency tone, make sure that tendency doesn't occur
    #   in other voices
    if resolution_chord.get_pitch_tendency(resolution_pitch) is not None:
        if resolution_pitch % 12 in {p % 12 for p in other_pitches}:
            return False

    # 2: make sure resolution pitch isn't doubled by melody, unless the melody
    #   is moving obliquely
    if melody_pitch is not None:
        if melody_pitch % 12 == resolution_pitch % 12:
            if prev_melody_pitch is None or prev_melody_pitch != melody_pitch:
                return False

    return True


SuspensionCombo = dict[Voice, Suspension]
