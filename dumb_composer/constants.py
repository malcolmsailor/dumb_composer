from fractions import Fraction
from types import MappingProxyType

from mspell import Speller, Unspeller

unspeller_pcs = Unspeller(pitches=False)
speller_pcs = Speller(pitches=False)
speller = Speller(pitches=True)
unspeller = Unspeller(pitches=True)

LOW_PITCH = 21
HI_PITCH = 108

MAJOR_TRIAD = (0, 4, 7)
MINOR_TRIAD = (0, 3, 7)
DIM_TRIAD = (0, 3, 6)
AUG_TRIAD = (0, 4, 8)

DOM_7TH = (0, 4, 7, 10)
MIN_7TH = (0, 3, 7, 10)
HALF_DIM = (0, 3, 6, 10)

DEFAULT_BASS_RANGE = (36, 50)
DEFAULT_MEL_RANGE = (60, 78)
DEFAULT_ACCOMP_RANGE = (49, 76)
DEFAULT_TENOR_MEL_RANGE = (48, 67)
DEFAULT_TENOR_ACCOMP_RANGE = (60, 84)

# TODO use some of these

DISSONANT_INTERVALS_ABOVE_BASS = {1, 2, 5, 6, 10, 11}
DISSONANT_INTERVAL_CLASSES_BETWEEN_UPPER_VOICES = {1, 2}
DISSONANT_INTERVALS_BETWEEN_UPPER_VOICES: set[
    int
] = DISSONANT_INTERVAL_CLASSES_BETWEEN_UPPER_VOICES | {
    12 - ic for ic in DISSONANT_INTERVAL_CLASSES_BETWEEN_UPPER_VOICES
}

CLOSE_REGISTERS = MappingProxyType(
    unspeller(  # type:ignore
        {
            0: ("Bb3", "F5"),
            1: ("Eb4", "C6"),
            2: ("Bb4", "G6"),
            3: ("F5", "C7"),
        }
    )
)
OPEN_REGISTERS = MappingProxyType(
    unspeller(  # type:ignore
        {
            0: ("A1", "E5"),
            1: ("A2", "C6"),
            2: ("D3", "G6"),
            3: ("F3", "C7"),
        }
    )
)
KEYBOARD_STYLE_REGISTERS = MappingProxyType(
    unspeller(  # type:ignore
        {
            0: (("A2", "C4"), ("D4", "E5")),
            1: (("D3", "F4"), ("G4", "A5")),
            2: (("F3", "A4"), ("A4", "C6")),
            3: (("C4", "C5"), ("E5", "A6")),
        }
    )
)

##################
# internal stuff #
##################

TIME_TYPE = Fraction
METER_CONDITIONS = (
    "triple",
    "duple",
    "compound",
)
TET = 12


OCTAVE_UNISON_WEIGHT = 0.0
PERFECT_CONSONANCE_WEIGHT = 0.0
IMPERFECT_CONSONANCE_WEIGHT = 5.0
DISSONANCE_WEIGHT = 2.5

# it would be nice to use MappingProxyType here except that it isn't pickleable
TWELVE_TET_HARMONIC_INTERVAL_WEIGHTS = {
    0: OCTAVE_UNISON_WEIGHT,
    1: DISSONANCE_WEIGHT,
    2: DISSONANCE_WEIGHT,
    3: IMPERFECT_CONSONANCE_WEIGHT,
    4: IMPERFECT_CONSONANCE_WEIGHT,
    5: DISSONANCE_WEIGHT,
    6: DISSONANCE_WEIGHT,
    7: PERFECT_CONSONANCE_WEIGHT,
    8: IMPERFECT_CONSONANCE_WEIGHT,
    9: IMPERFECT_CONSONANCE_WEIGHT,
    10: DISSONANCE_WEIGHT,
    11: DISSONANCE_WEIGHT,
    12: OCTAVE_UNISON_WEIGHT,
}

TWELVE_TET_SUSPENSION_RESOLUTION_INTERVAL_WEIGHTS = {
    0: 0.0,
    1: 1.0,
    2: 1.0,
    3: 1.5,
    4: 1.5,
    5: 1.0,
    6: 1.0,
    7: 0.75,
    8: 1.5,
    9: 1.5,
    10: 1.0,
    11: 1.0,
    12: 0.75,
}
