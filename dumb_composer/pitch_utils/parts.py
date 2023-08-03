import typing as t
from itertools import chain, combinations, count, product

import numpy as np

from dumb_composer.constants import MAX_TIME, unspeller
from dumb_composer.pitch_utils.intervals import reduce_compound_interval
from dumb_composer.pitch_utils.types import (
    ChromaticInterval,
    MelodicAtom,
    NoteChange,
    Pitch,
    Simultaneity,
    SimultaneousNoteChange,
    TimeStamp,
    VoiceAssignments,
)
from dumb_composer.shared_classes import Note

T = t.TypeVar("T")

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
    >>> infer_voice_assignments(
    ...     [
    ...         60,
    ...     ],
    ...     [59, 67],
    ... )
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
    ...     [60, 64, 67],
    ...     [59, 62, 65],
    ...     src_assignments=(0, 1, 2),
    ...     dst_assignments=(0, 1, 2),
    ... )
    [(60, 59), (64, 62), (67, 65)]

    An index occurs once on one voice and more than once in the other:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67],
    ...     [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 2),
    ...     dst_assignments=(0, 1, 1, 2),
    ... )
    [(60, 59), (64, 62), (64, 65), (67, 67)]

    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67, 70, 72],
    ...     [60, 65, 69],
    ...     src_assignments=(0, 1, 1, 1, 2),
    ...     dst_assignments=(0, 1, 2),
    ... )
    [(60, 60), (64, 65), (67, 65), (70, 65), (72, 69)]

    An index occurs in one voice and not the other:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67],
    ...     [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 3),
    ...     dst_assignments=(0, 1, 2, 4),
    ... )
    [(60, 59), (64, 62), (None, 65), (67, None), (None, 67)]

    ------------------------------------------------------------------------------------
    Edge cases
    ------------------------------------------------------------------------------------

    `src_pitches` is empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [], [59, 62, 65, 67], src_assignments=(), dst_assignments=(0, 1, 1, 3)
    ... )
    [(None, 59), (None, 62), (None, 65), (None, 67)]

    `dst_pitches` is empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [62, 65, 67], [], src_assignments=(2, 5, 11), dst_assignments=()
    ... )
    [(62, None), (65, None), (67, None)]

    Both are empty:
    >>> chord_succession_to_melodic_atoms(
    ...     [], [], src_assignments=(), dst_assignments=()
    ... )
    []

    "Missing" indices do not cause an error, however, they have no effect:
    >>> chord_succession_to_melodic_atoms(
    ...     [60, 64, 67],
    ...     [59, 62, 65, 67],
    ...     src_assignments=(0, 1, 5),
    ...     dst_assignments=(0, 1, 1, 5),
    ... )
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

    >>> succession_has_forbidden_parallels(
    ...     [60, 72],
    ...     [53, 77],
    ...     forbidden_antiparallels=[
    ...         12,
    ...     ],
    ... )
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


def quick_notes(note_str: str, increment: TimeStamp = TimeStamp(1.0)) -> list[Note]:
    """
    For use in docstrings

    The syntax is as follows:
    - each string consists of white-space separated "atoms" of three types
        - spelled pitches: indicates the onset of a note
        - "-": indicates the continuation of the already-sounding note
        - ".": indicates rest

    Each atom is isochronous, having the duration indicated by `timestamp`

    >>> quick_notes("F4  G4  A4")
    [65:0-1, 67:1-2, 69:2-3]
    >>> quick_notes("F4  -   A4")
    [65:0-2, 69:2-3]
    >>> quick_notes("F4  .   A4")
    [65:0-1, 69:2-3]
    >>> quick_notes(".   G4  A4")
    [67:1-2, 69:2-3]

    An initial "-" or "-" following a rest will raise a ValueError
    >>> quick_notes("-   G4  A4")
    Traceback (most recent call last):
    ValueError
    >>> quick_notes(".   -   A4")
    Traceback (most recent call last):
    ValueError
    """
    atoms = note_str.split()
    out = []
    now = TimeStamp(0.0)
    pitch: Pitch | None = None
    note_start: TimeStamp | None = None
    for atom in atoms:
        if atom == "-":
            now += increment
            if pitch is None:
                raise ValueError()
            continue
        if pitch is not None:
            out.append(Note(pitch, onset=note_start, release=now))  # type:ignore
        if atom == ".":
            pitch = None
            note_start = None
        else:
            pitch = unspeller(atom)
            note_start = now
        now += increment
    if pitch is not None:
        out.append(Note(pitch, onset=note_start, release=now))  # type:ignore
    return out


class NotePeeker:
    """
    >>> notes = quick_notes("F4  -  .  G4  A4  .  ")
    >>> note_peeker = NotePeeker(notes)
    >>> note_peeker.current_note, note_peeker.next_note
    (65:0-2, 67:3-4)
    >>> note_peeker.advance_to(1.0)
    >>> note_peeker.current_note, note_peeker.next_note
    (65:0-2, 67:3-4)
    >>> note_peeker.advance_to(1.99)
    >>> note_peeker.current_note, note_peeker.next_note
    (65:0-2, 67:3-4)
    >>> note_peeker.advance_to(2.0)
    >>> note_peeker.current_note, note_peeker.next_note
    (None, 67:3-4)
    >>> note_peeker.advance_to(2.99)
    >>> note_peeker.current_note, note_peeker.next_note
    (None, 67:3-4)
    >>> note_peeker.advance_to(3.0)
    >>> note_peeker.current_note, note_peeker.next_note
    (67:3-4, 69:4-5)
    >>> note_peeker.advance_to(4.5)
    >>> note_peeker.current_note, note_peeker.next_note
    (69:4-5, None)
    >>> note_peeker.advance_to(5.0)
    >>> note_peeker.current_note, note_peeker.next_note
    (None, None)

    >>> note_peeker.advance_to(4.0)
    Traceback (most recent call last):
    ValueError


    >>> notes = quick_notes(". F4  -  .  G4  ")
    >>> note_peeker = NotePeeker(notes)
    >>> note_peeker.current_note, note_peeker.next_note
    (None, 65:1-3)
    >>> note_peeker.advance_to(1.0)
    >>> note_peeker.current_note, note_peeker.next_note
    (65:1-3, 67:4-5)
    """

    def __init__(self, seq: t.Sequence[Note]):
        self._now = 0.0
        self._current_note: Note | None = None
        self._next_note: Note | None = None if not seq else seq[0]
        self._iter = iter(seq[1:])
        self.advance_to(TimeStamp(0.0))

    @property
    def current_note(self):
        return self._current_note

    @property
    def next_note(self):
        return self._next_note

    def advance_to(self, time: TimeStamp):
        if time < self._now:
            raise ValueError()
        self._now = time
        if self._current_note is not None and time >= self._current_note.release:
            self._current_note = None
        if self._next_note is not None and time >= self._next_note.onset:
            self._current_note = self._next_note
            try:
                self._next_note = next(self._iter)
            except StopIteration:
                self._next_note = None

    @property
    def next_event(self):
        if self.current_note is not None:
            if self.current_note.onset > self._now:
                return self.current_note.onset
            if self.current_note.release > self._now:
                return self.current_note.release
        if self.next_note is not None:
            if self.next_note.onset > self._now:
                return self.next_note.onset
            if self.next_note.release > self._now:
                return self.next_note.release
        return MAX_TIME


def get_simultaneities(notes: t.Sequence[t.Sequence[Note]]) -> list[Simultaneity]:
    """
    >>> upper_voice = quick_notes("F4  G4  -   A4")
    >>> lower_voice = quick_notes("Bb3 -   C4  F3")
    >>> for simultaneity in get_simultaneities((lower_voice, upper_voice)):
    ...     print(simultaneity)
    ...
    [58:0-1⌒, 65:0-1]
    [⌒58:1-2, 67:1-2⌒]
    [60:2-3, ⌒67:2-3]
    [53:3-4, 69:3-4]

    >>> upper_voice = quick_notes(".   G4  .   A4")
    >>> lower_voice = quick_notes("Bb3 .   C4  .")
    >>> for simultaneity in get_simultaneities((lower_voice, upper_voice)):
    ...     print(simultaneity)
    ...
    [58:0-1, None]
    [None, 67:1-2]
    [60:2-3, None]
    [None, 69:3-4]

    >>> upper_voice = quick_notes("G4  -   -   -")
    >>> lower_voice = quick_notes("Bb3 .   .   C4")
    >>> for simultaneity in get_simultaneities((lower_voice, upper_voice)):
    ...     print(simultaneity)
    ...
    [58:0-1, 67:0-1⌒]
    [None, ⌒67:1-3⌒]
    [60:3-4, ⌒67:3-4]
    """

    note_peekers = [NotePeeker(n) for n in notes]

    current_time = -1
    out: list[Simultaneity] = []

    next_time = TimeStamp(0.0)
    tie_to_prev_array = [False for _ in note_peekers]

    while next_time < MAX_TIME:
        if current_time >= 0:
            simultaneity: list[Note | None] = []
            for note_peeker_i, note_peeker in enumerate(note_peekers):
                note = note_peeker.current_note
                if note is None:
                    simultaneity.append(None)
                    continue

                tie_to_next = note.tie_to_next or note.release > next_time
                note_copy = note.partial_copy(
                    onset=current_time,
                    release=next_time,
                    tie_to_next=tie_to_next,
                    tie_to_prev=tie_to_prev_array[note_peeker_i],
                )
                tie_to_prev_array[note_peeker_i] = tie_to_next
                simultaneity.append(note_copy)
            out.append(simultaneity)
        for note_peeker in note_peekers:
            note_peeker.advance_to(next_time)
        current_time = next_time
        next_time = min(note_peeker.next_event for note_peeker in note_peekers)

    return out


def get_simultaneous_note_changes(
    simultaneities: t.Sequence[Simultaneity], min_n_onsets: int = 2
) -> list[SimultaneousNoteChange]:
    """
    >>> upper_voice = quick_notes("F4  G4  -   A4")
    >>> lower_voice = quick_notes("Bb3 -   C4  F3")
    >>> simultaneities = get_simultaneities((lower_voice, upper_voice))
    >>> get_simultaneous_note_changes(simultaneities)
    [((60:2-3, 53:3-4), (⌒67:2-3, 69:3-4))]

    >>> upper_voice = quick_notes("F4  G4  -   A4")
    >>> inner_voice = quick_notes("D4  D4  E4  C4")
    >>> lower_voice = quick_notes("Bb3 -   C4  F3")
    >>> simultaneities = get_simultaneities((lower_voice, inner_voice, upper_voice))
    >>> get_simultaneous_note_changes(simultaneities)  # doctest: +NORMALIZE_WHITESPACE
    [((62:0-1,  62:1-2), (65:0-1, 67:1-2⌒)),
     ((⌒58:1-2, 60:2-3), (62:1-2, 64:2-3)),
     ((60:2-3,  53:3-4), (64:2-3, 60:3-4), (⌒67:2-3, 69:3-4))]
    """
    out: list[tuple[NoteChange, ...]] = []
    for simultaneity1, simultaneity2 in zip(simultaneities[:-1], simultaneities[1:]):
        note_changes = [
            (note1, note2)
            for (note1, note2) in zip(simultaneity1, simultaneity2)
            if note2 is not None and not note2.tie_to_prev
        ]
        if len(note_changes) < min_n_onsets:
            continue
        out.append(tuple(note_changes))
    return out


def get_pairs(items: t.Sequence[t.Sequence[T]]) -> t.Iterator[tuple[T, T]]:
    for group in items:
        yield from combinations(group, 2)


def simultaneous_note_changes_have_forbidden_parallels(
    note_changes: list[SimultaneousNoteChange],
    forbidden_parallel_intervals: t.Container[ChromaticInterval],
) -> bool:
    for change1, change2 in get_pairs(note_changes):
        if None in change1 or None in change2:
            continue
        first_interval = change1[0].pitch - change2[0].pitch  # type:ignore
        if (
            abs(reduce_compound_interval(first_interval))
            in forbidden_parallel_intervals
        ):
            if change1[1].pitch - change2[1].pitch == first_interval:  # type:ignore
                return True
    return False


def note_lists_have_forbidden_parallels(
    note_list1: list[Note],
    note_list2: list[Note],
    forbidden_parallel_intervals: t.Container[ChromaticInterval] = frozenset(
        {0, 7, 12}
    ),
) -> bool:
    """
    ------------------------------------------------------------------------------------
    Straightforward cases
    ------------------------------------------------------------------------------------

    1. The most straightforward cases are when both voices have simultaneous onsets from a
    previously sounding note:

    Metric weight:  s w s
    Soprano:        F   G
    Bass:           F   G
    >>> upper_voice = quick_notes("F4  -   G4")
    >>> lower_voice = quick_notes("F3  -   G3")
    >>> note_lists_have_forbidden_parallels(upper_voice, lower_voice)
    True

    Metric weight:  s w s
    Soprano:        F   G
    Bass:           . F G
    >>> upper_voice = quick_notes("F4  -   G4")
    >>> lower_voice = quick_notes(".   F3  G3")
    >>> note_lists_have_forbidden_parallels(upper_voice, lower_voice)
    True

    # TODO: (Malcolm 2023-07-31) so far this function takes no account of metric weight
    Metric weight:  s w s w
    Soprano:        . F   G
    Bass:           . F   G

    2. A little more difficult is when there is an interceding rest in one or both voices:

    # TODO: (Malcolm 2023-07-31) this and the following not implemented
    Metric weight:  s w s
    Bass:           F   G
    Soprano:        F . G
    >>> upper_voice = quick_notes("F4  -   G4")
    >>> lower_voice = quick_notes("F3  .   G3")
    >>> note_lists_have_forbidden_parallels(upper_voice, lower_voice)  # doctest: +SKIP

    True
    Metric weight:  s w s
    Soprano:        F . G
    Bass:           F . G

    Metric weight:  s w s w
    Soprano:        . F . G
    Bass:           . F . G

    If these interceding rests are brief then these seem like straightforward parallels.
    We could treat these in any of the following ways:
        A. ignore if there is one or more interceding harmonies during the rest. For
            this we require a harmonic analysis, however.
        B. ignore if there is a section boundary between the parallels (e.g., the end of
            the minuet and beginning of the trio). It isn't clear how often we will
            have such information, however.
        C. ignore if the rest is longer than some threshold length (or if the rest + the
            preceding note is longer than some threshold length).

    ------------------------------------------------------------------------------------
    More difficult cases
    ------------------------------------------------------------------------------------

    1. When there is a rest simultaneous with the onset of one of the notes:

    Metric weight:  s w s w
    Soprano:        F   G
    Bass:           F   . G

    Metric weight:  s w s w
    Soprano:        F G
    Bass:           F . G

    Metric weight:  s w s w
    Soprano:        F   G
    Bass:           . F . G

    2. When the notes never sound simultaneously:

    Metric weight:  s w s w
    Soprano:        F . G .
    Bass:           . F . G

    3. When there is an intervening note:

    A. intervening consonance:

    Metric weight:  s w s
    Soprano:        F   G
    Bass:           F A G

    B. intervening dissonance:

    Metric weight:  s w s
    Soprano:        F   G
    Bass:           F E G

    C. intervening chromatic passing tone:

    Metric weight:  s w  s
    Soprano:        F    G
    Bass:           F F# G

    D. intervening anticipation:

    Metric weight:  s w s
    Soprano:        F F G
    Bass:           F   G


    """
    simultaneities = get_simultaneities([note_list1, note_list2])
    note_changes = get_simultaneous_note_changes(simultaneities)
    return simultaneous_note_changes_have_forbidden_parallels(
        note_changes, forbidden_parallel_intervals
    )
