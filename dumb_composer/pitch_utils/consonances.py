import typing as t
from dumb_composer.pitch_utils.scale import Scale
from dumb_composer.pitch_utils.pcs import quasi_pitch_order
import itertools as it

GENERIC_INTERVAL_TO_CHROMATIC_INTERVAL_BASS = {
    0: (0,),
    2: (3, 4),
    4: (7,),
    5: (8, 9),
}

GENERIC_INTERVAL_TO_CHROMATIC_INTERVAL_UPPER_VOICES = {
    0: (0,),
    2: (3, 4),
    3: (5, 6),
    4: (6, 7),
    5: (8, 9),
}


def pair_of_pitches_consonant(
    pitch1: int,
    pitch2: int,
    scale: t.Union[Scale, t.Sequence[int]],
    pitch1_is_bass: bool = False,
) -> bool:
    """
    pitch1 is assumed to be <= pitch2

    >>> c_major = (0, 2, 4, 5, 7, 9, 11)
    >>> pair_of_pitches_consonant(60, 64, c_major)
    True
    >>> pair_of_pitches_consonant(60, 65, c_major)
    True
    >>> pair_of_pitches_consonant(60, 65, c_major, pitch1_is_bass=True)
    False

    If any of the pitch-classes are not in the scale, we rely on Scale.get_interval()
    to infer the generic interval. This won't necessarily always gi ve the desired
    result.
    >>> pair_of_pitches_consonant(61, 64, c_major)
    True
    >>> pair_of_pitches_consonant(61, 64, (0, 1, 4, 5, 7, 8, 10)) # F harmonic minor
    False

    >>> pair_of_pitches_consonant(60, 72, c_major)
    True
    >>> pair_of_pitches_consonant(61, 61, c_major)
    True
    >>> pair_of_pitches_consonant(61, 73, c_major)
    True
    >>> pair_of_pitches_consonant(60, 61, c_major)
    False

    >>> pair_of_pitches_consonant(61, 60, c_major)
    Traceback (most recent call last):
    AssertionError
    """
    assert pitch1 <= pitch2

    if not isinstance(scale, Scale):
        scale = Scale(scale)

    generic_interval_mapping = (
        GENERIC_INTERVAL_TO_CHROMATIC_INTERVAL_BASS
        if pitch1_is_bass
        else GENERIC_INTERVAL_TO_CHROMATIC_INTERVAL_UPPER_VOICES
    )

    generic_interval = scale.get_interval(pitch1, pitch2, reduce_compounds=True)

    if generic_interval not in generic_interval_mapping:
        return False

    chromatic_interval = (pitch2 - pitch1) % 12

    return chromatic_interval in generic_interval_mapping[generic_interval]


def pitches_consonant(
    pitches: t.Sequence[int],
    scale: t.Union[Scale, t.Sequence[int]],
    first_pitch_is_bass: bool = True,
) -> bool:
    """
    pitches should be sorted

    >>> c_major = (0, 2, 4, 5, 7, 9, 11)
    >>> pitches_consonant((60, 64, 67), c_major)
    True
    >>> pitches_consonant((55, 60, 64), c_major)
    False
    >>> pitches_consonant((55, 60, 64), c_major, first_pitch_is_bass=False)
    True
    >>> pitches_consonant((60, 69), c_major)
    True
    >>> pitches_consonant((60, 69), c_major, first_pitch_is_bass=False)
    True

    >>> pitches_consonant((62, 65, 71), c_major)
    True

    >>> c_minor = (0, 2, 3, 5, 7, 8, 11)
    >>> pitches_consonant((64, 68, 72), c_minor)
    False

    >>> pitches_consonant((60, 64, 55), c_major)
    Traceback (most recent call last):
    AssertionError
    """
    if len(pitches) < 2:
        return True

    if not isinstance(scale, Scale):
        scale = Scale(scale)

    if first_pitch_is_bass:
        non_bass_pitches = pitches[1:]
    else:
        non_bass_pitches = pitches

    if first_pitch_is_bass:
        for upper_pitch in non_bass_pitches:
            if not pair_of_pitches_consonant(
                pitches[0], upper_pitch, scale, pitch1_is_bass=True
            ):
                return False

    for pitch1, pitch2 in it.combinations(non_bass_pitches, r=2):
        if not pair_of_pitches_consonant(pitch1, pitch2, scale):
            return False
    return True


def pcs_consonant(
    pcs: t.Sequence[int],
    scale: t.Union[Scale, t.Sequence[int]],
    first_pitch_is_bass: bool = True,
) -> bool:
    """pcs are assumed to be in "ascending pitch order". I.e., if we receive
        (7, 4, 0)
    We assume:
        - the pitch realizing pc 4 > pitch realizing pc 7
        - the pitch realizing pc 0 > pitch realizing pc 4
    So (G4, E5, C6) is a possible realization of these pcs but not (G4, E5, C5).

    >>> c_major = (0, 2, 4, 5, 7, 9, 11)
    >>> pcs_consonant((0, 4, 7), c_major)
    True
    >>> pcs_consonant((7, 0, 4), c_major)
    False
    >>> pcs_consonant((7, 0, 4), c_major, first_pitch_is_bass=False)
    True
    """
    # We first convert to "quasi-pitches" so that pcs are in "ascending pitch order"
    quasi_pitches = quasi_pitch_order(pcs)
    return pitches_consonant(quasi_pitches, scale, first_pitch_is_bass)
