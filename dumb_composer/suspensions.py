from dataclasses import dataclass
import typing as t

from dumb_composer.constants import (
    DISSONANT_INTERVALS_ABOVE_BASS,
    DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES,
)


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
    score: float = 1.0


def find_suspensions(
    current_pitch: int,
    next_chord_pcs: t.Sequence[int],
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

    If the current pitch is already in the next chord, it can't be a
    suspension.

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
    """

    def _append(current_pitch, interval, next_pc):
        interval_above_bass = (current_pitch - next_chord_pcs[0]) % 12
        other_pcs = list(next_chord_pcs)
        other_pcs.remove(next_pc)
        dissonant = pitch_dissonant_against_chord(current_pitch, other_pcs)
        out.append(Suspension(interval, dissonant, interval_above_bass))

    out = []
    next_chord_set = set(next_chord_pcs)
    if current_pitch % 12 in next_chord_set:
        return out
    for interval in resolve_down_by + resolve_up_by:
        if (next_pc := (current_pitch + interval) % 12) in next_chord_set:
            _append(current_pitch, interval, next_pc)
    return out
