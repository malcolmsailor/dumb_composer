from itertools import combinations, count
import typing as t

import numpy as np
from dumb_composer.pitch_utils.intervals import reduce_compound_interval

from dumb_composer.pitch_utils.types import (
    ChromaticInterval,
    MelodicAtom,
    Pitch,
    VoiceAssignments,
)


# TODO: (Malcolm) write a smarter version of `infer_voice_assignments`
"""
Spec for smarter infer_voice_assignments():
If number of pitches agrees, return solution immediately
If number of pitches differs
- if there is only 1 voice in one of the chords, map it to the closer of the 
    bass/melody voice and assign all other voices accordingly
- if there is only 2 voices in one of the chords, map them to bass/melody and assign
    inner voices of other chord accordingly
- if there are >= 3 voices in both chords, find mapping s.t.,
    - every voice is mapped at least once
    - total displacement is least
    - we can maybe use voice-leading function to do this
"""


def infer_voice_assignments(
    src_pitches: t.Sequence[Pitch], dst_pitches: t.Sequence[Pitch]
) -> t.Tuple[VoiceAssignments, VoiceAssignments]:
    """
    This is a very "dumb" function.
        - src_assignments = range(len(src_pitches))
        - dst_assignments gets:
            - 0 for the bass
            - the highest voice from src_assignments for the melody (unless this is 0,
                in which case we assign 1)
            - for any inner voices in dst_pitches, we simply assign inner voices from
                src_assignments until we exhaust them, then assign the last voice
                repeatedly.

    Note: we take no account of proximity in making voice assignments.

    >>> infer_voice_assignments([60, 64, 67], [59, 62, 67])
    ((0, 1, 2), (0, 1, 2))

    >>> infer_voice_assignments([60, 64, 67], [59, 62, 65, 68])
    ((0, 1, 2), (0, 1, 1, 2))

    >>> infer_voice_assignments([60, 64], [59, 62, 67])
    ((0, 1), (0, 0, 1))

    Exception: we always add a second voice for melody
    >>> infer_voice_assignments([60,], [59, 67])
    ((0,), (0, 1))

    >>> infer_voice_assignments([60, 67], [59])
    ((0, 1), (0,))

    >>> infer_voice_assignments([60, 64, 67, 70], [53, 65, 69])
    ((0, 1, 2, 3), (0, 1, 3))

    >>> infer_voice_assignments([60, 64, 67], [48, 52, 55, 60, 64, 67, 72])
    ((0, 1, 2), (0, 1, 1, 1, 1, 1, 2))

    >>> infer_voice_assignments([48, 52, 55, 60, 64, 67, 72], [60, 64, 67])
    ((0, 1, 2, 3, 4, 5, 6), (0, 1, 6))

    >>> infer_voice_assignments([60, 64, 67], [])
    ((0, 1, 2), ())

    >>> infer_voice_assignments([], [59, 62, 67])
    ((), (0, 1, 2))

    """
    src_assignments = tuple(range(len(src_pitches)))

    if not src_assignments:
        return src_assignments, tuple(range(len(dst_pitches)))

    if not dst_pitches:
        return src_assignments, ()

    # Add bass voice first
    dst_assignments = [0]

    # If inner voices exist, add them
    assignment_i = 0
    assignment_iter = iter(range(1, len(src_assignments) - 1))
    for inner_pitch in dst_pitches[1:-1]:
        try:
            assignment_i = next(assignment_iter)
        except StopIteration:
            pass
        dst_assignments.append(assignment_i)

    # If melody voice exists, add it
    if len(dst_pitches) > 1:
        dst_assignments.append(max(len(src_pitches) - 1, 1))

    return src_assignments, tuple(dst_assignments)


def chord_succession_to_melodic_atoms(
    src_pitches: t.Sequence[Pitch],
    dst_pitches: t.Sequence[Pitch],
    src_assignments: VoiceAssignments,
    dst_assignments: VoiceAssignments,
) -> t.List[MelodicAtom]:
    """

    Assumes:
    - voice assignments are sorted in ascending order.
    - len(src_pitches) == len(src_assignments)
    - len(dst_pitches) == len(dst_assignments)

    Voice assignments are valid if each index occurs at most once in at least one voice.
    Thus the following voice assignment types are valid:
    1. the index occurs once in both voices:
        (1,) -> (1,)
    2. the index occurs once in one voice and more than once in the other:
        (1,) -> (1, 1)
        (1, 1) -> (1,)
    3. the index does not occur in one voice and occurs one or more times in the other:
        () -> (1,)
        () -> (1, 1)

    ------------------------------------------------------------------------------------
    Test cases
    ------------------------------------------------------------------------------------

    All indices occur once in both voices:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67], [59, 62, 65],
    ...     src_assignments=(0, 1, 2),
    ...     dst_assignments=(0, 1, 2))
    [(60, 59), (64, 62), (67, 65)]

    An index occurs once on one voice and more than once in the other:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67], [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 2),
    ...     dst_assignments=(0, 1, 1, 2))
    [(60, 59), (64, 62), (64, 65), (67, 67)]

    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67, 70, 72], [60, 65, 69],
    ...     src_assignments=(0, 1, 1, 1, 2),
    ...     dst_assignments=(0, 1, 2))
    [(60, 60), (64, 65), (67, 65), (70, 65), (72, 69)]

    An index occurs in one voice and not the other:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67], [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 3),
    ...     dst_assignments=(0, 1, 2, 4))
    [(60, 59), (64, 62), (None, 65), (67, None), (None, 67)]

    ------------------------------------------------------------------------------------
    Edge cases
    ------------------------------------------------------------------------------------

    `src_pitches` is empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [], [59, 62, 65, 67],
    ...     src_assignments=(),
    ...     dst_assignments=(0, 1, 1, 3))
    [(None, 59), (None, 62), (None, 65), (None, 67)]

    `dst_pitches` is empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [62, 65, 67], [],
    ...     src_assignments=(2, 5, 11),
    ...     dst_assignments=())
    [(62, None), (65, None), (67, None)]

    Both are empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [], [],
    ...     src_assignments=(),
    ...     dst_assignments=())
    []

    "Missing" indices do not cause an error, however, they have no effect:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67], [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 5),
    ...     dst_assignments=(0, 1, 1, 5))
    [(60, 59), (64, 62), (64, 65), (67, 67)]
    """

    if not src_pitches:
        return [(None, p) for p in dst_pitches]
    if not dst_pitches:
        return [(p, None) for p in src_pitches]

    out = []

    # -----------------------------------------------------------------------------------
    # Initialize src items
    # -----------------------------------------------------------------------------------

    src_i = 0
    next_src_i = 1 if len(src_pitches) > 1 else None
    src_iter = iter(src_assignments[2:])
    chord1_i = 0

    def _advance_src():
        nonlocal src_i, next_src_i, chord1_i
        src_i = next_src_i
        try:
            next_src_i = next(src_iter)
        except StopIteration:
            next_src_i = None
        chord1_i += 1

    # -----------------------------------------------------------------------------------
    # Initialize dst items
    # -----------------------------------------------------------------------------------

    dst_i = 0
    next_dst_i = 1 if len(dst_pitches) > 1 else None
    dst_iter = iter(dst_assignments[2:])
    chord2_i = 0

    def _advance_dst():
        nonlocal dst_i, next_dst_i, chord2_i
        dst_i = next_dst_i
        try:
            next_dst_i = next(dst_iter)
        except StopIteration:
            next_dst_i = None
        chord2_i += 1

    # -----------------------------------------------------------------------------------
    # Run loop
    # -----------------------------------------------------------------------------------

    for i in count():
        while src_i == i or dst_i == i:
            if src_i == dst_i:
                out.append((src_pitches[chord1_i], dst_pitches[chord2_i]))

                # We need to check whether we need to advance dst before possibly
                # advancing src
                dst_must_be_advanced = next_src_i != i

                if next_dst_i != i:
                    _advance_src()
                if dst_must_be_advanced:
                    _advance_dst()
            elif src_i == i:
                out.append((src_pitches[chord1_i], None))
                _advance_src()
            else:
                assert dst_i == i
                out.append((None, dst_pitches[chord2_i]))
                _advance_dst()
        if dst_i is None and src_i is None:
            break

    return out


# TODO: (Malcolm) allow providing voice assignments, compare results
def succession_has_forbidden_parallels(
    src_pitches: t.Sequence[Pitch],
    dst_pitches: t.Sequence[Pitch],
    forbidden_parallels: t.Sequence[ChromaticInterval] = (0, 7, 12),
    forbidden_antiparallels: t.Sequence[ChromaticInterval] = (),
) -> bool:
    """
    Assumes that src_pitches and dst_pitches are each in sorted order.

    ------------------------------------------------------------------------------------
    Octaves
    ------------------------------------------------------------------------------------
    >>> succession_has_forbidden_parallels([60, 72], [62, 74])
    True
    >>> succession_has_forbidden_parallels([60, 72], [59, 71])
    True
    >>> succession_has_forbidden_parallels([60, 72], [60, 72])
    False
    >>> succession_has_forbidden_parallels([60, 72], [48, 72])
    False
    >>> succession_has_forbidden_parallels([60, 72], [60, 84])
    False

    Compound octaves:
    >>> succession_has_forbidden_parallels([48, 72], [50, 74])
    True

    Outer voices:
    >>> succession_has_forbidden_parallels([60, 64, 67, 72], [62, 65, 74])
    True
    >>> succession_has_forbidden_parallels([60, 64, 72], [62, 65, 69, 74])
    True

    Note that "quasi-octaves by similar motion" (e.g., an octave followed by a 15th by
    similar motion) are treated as parallels (because we reduce compound intervals):
    >>> succession_has_forbidden_parallels([48, 72], [62, 74])
    True
    >>> succession_has_forbidden_parallels([60, 72], [46, 70])
    True

    ------------------------------------------------------------------------------------
    Fifths
    ------------------------------------------------------------------------------------
    >>> succession_has_forbidden_parallels([60, 67], [62, 69])
    True
    >>> succession_has_forbidden_parallels([60, 67], [60, 67])
    False
    >>> succession_has_forbidden_parallels([60, 65], [62, 67])
    False

    ------------------------------------------------------------------------------------
    Unisons
    ------------------------------------------------------------------------------------
    >>> succession_has_forbidden_parallels([60, 60], [62, 62])
    True
    >>> succession_has_forbidden_parallels([60, 60], [60, 60])
    False

    ------------------------------------------------------------------------------------
    Antiparallels
    ------------------------------------------------------------------------------------
    By default, no antiparallels are forbidden:
    >>> succession_has_forbidden_parallels([60, 72], [53, 77])
    False

    >>> succession_has_forbidden_parallels([60, 72], [53, 77], forbidden_antiparallels=[12,])
    True
    """
    src_assignments, dst_assignments = infer_voice_assignments(src_pitches, dst_pitches)
    melodic_atoms = chord_succession_to_melodic_atoms(
        src_pitches, dst_pitches, src_assignments, dst_assignments
    )

    for (atom1_p1, atom1_p2), (atom2_p1, atom2_p2) in combinations(melodic_atoms, r=2):
        # If any note is missing, continue
        if atom1_p1 is None or atom1_p2 is None or atom2_p1 is None or atom2_p2 is None:
            continue

        melodic_interval1 = atom1_p2 - atom1_p1
        melodic_interval2 = atom2_p2 - atom2_p1

        # If either voice has a melodic unison, continue
        if melodic_interval1 == 0 or melodic_interval2 == 0:
            continue

        # If sign of melodic intervals does not agree, check antiparallels
        if (melodic_interval1 > 0) != (melodic_interval2 > 0):
            forbidden_intervals = forbidden_antiparallels
        else:
            forbidden_intervals = forbidden_parallels
        if not forbidden_intervals:
            continue

        harmonic_interval1 = reduce_compound_interval(abs(atom2_p1 - atom1_p1))
        harmonic_interval2 = reduce_compound_interval(abs(atom2_p2 - atom1_p2))

        if (
            harmonic_interval1 == harmonic_interval2
            and harmonic_interval1 in forbidden_intervals
        ):
            return True

    return False


def outer_voices_have_forbidden_antiparallels(
    src_pitches: t.Sequence[Pitch],
    dst_pitches: t.Sequence[Pitch],
    forbidden_antiparallels: t.Sequence[ChromaticInterval] = (12,),
) -> bool:
    """
    Generally, we want to check for forbidden antiparallels between the outer voices
    only, hence this function.

    >>> outer_voices_have_forbidden_antiparallels([60, 64, 67, 72], [53, 69, 77])
    True
    >>> outer_voices_have_forbidden_antiparallels([48, 64, 67, 72], [55, 59, 67])
    True
    """
    if len(src_pitches) < 2 or len(dst_pitches) < 2:
        return False

    bass_pitch1, bass_pitch2 = src_pitches[0], dst_pitches[0]
    melody_pitch1, melody_pitch2 = src_pitches[-1], dst_pitches[-1]
    bass_melodic_interval = bass_pitch2 - bass_pitch1
    melody_melodic_interval = melody_pitch2 - melody_pitch1

    # If either voice moves obliquely, we don't have antiparallels
    if bass_melodic_interval == 0 or melody_melodic_interval == 0:
        return False

    # If both melodic intervals have same sign, we don't have antiparallels
    if (bass_melodic_interval > 0) == (melody_melodic_interval > 0):
        return False

    harmonic_interval1 = reduce_compound_interval(abs(melody_pitch1 - bass_pitch1))
    harmonic_interval2 = reduce_compound_interval(abs(melody_pitch2 - bass_pitch2))

    return (
        harmonic_interval1 == harmonic_interval2
        and harmonic_interval1 in forbidden_antiparallels
    )
