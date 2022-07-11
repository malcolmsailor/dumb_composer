from fractions import Fraction
from types import MappingProxyType

from mspell import Unspeller

unspeller = Unspeller(pitches=True)

MAJOR_TRIAD = (0, 4, 7)
MINOR_TRIAD = (0, 3, 7)
DIM_TRIAD = (0, 3, 6)
AUG_TRIAD = (0, 4, 8)

DOM_7TH = (0, 4, 7, 10)
MIN_7TH = (0, 3, 7, 10)
HALF_DIM = (0, 3, 6, 10)

DEFAULT_RANGES = ((36, 60), (55, 84))

CLOSE_REGISTERS = MappingProxyType(
    unspeller(
        {
            0: ("Bb3", "F5"),
            1: ("Eb4", "C6"),
            2: ("Bb4", "G6"),
            3: ("F5", "C7"),
        }
    )
)
OPEN_REGISTERS = MappingProxyType(
    unspeller(
        {
            0: ("C3", "E5"),
            1: ("A3", "C6"),
            2: ("D4", "G6"),
            3: ("F4", "C7"),
        }
    )
)
KEYBOARD_STYLE_REGISTERS = MappingProxyType(
    unspeller(
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
