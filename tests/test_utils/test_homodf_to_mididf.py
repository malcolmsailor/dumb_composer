from fractions import Fraction

import pandas as pd
from dumb_composer.utils.homodf_to_mididf import homodf_to_mididf


def test_homodf_to_mididf():
    test_df = pd.DataFrame(
        {
            "onset": {
                0: Fraction(0, 1),
                1: Fraction(4, 1),
                2: Fraction(8, 1),
                3: Fraction(12, 1),
                4: Fraction(16, 1),
                5: Fraction(20, 1),
                6: Fraction(24, 1),
                7: Fraction(28, 1),
                8: Fraction(32, 1),
                9: Fraction(36, 1),
                10: Fraction(40, 1),
                11: Fraction(44, 1),
                12: Fraction(48, 1),
                13: Fraction(52, 1),
            },
            "release": {
                0: Fraction(4, 1),
                1: Fraction(8, 1),
                2: Fraction(12, 1),
                3: Fraction(16, 1),
                4: Fraction(20, 1),
                5: Fraction(24, 1),
                6: Fraction(28, 1),
                7: Fraction(32, 1),
                8: Fraction(36, 1),
                9: Fraction(40, 1),
                10: Fraction(44, 1),
                11: Fraction(48, 1),
                12: Fraction(52, 1),
                13: Fraction(56, 1),
            },
            "bass": {
                0: 34,
                1: 31,
                2: 36,
                3: 36,
                4: 41,
                5: 34,
                6: 36,
                7: 41,
                8: 36,
                9: 36,
                10: 41,
                11: 31,
                12: 34,
                13: 33,
            },
            "melody": {
                0: 65,
                1: 67,
                2: 69,
                3: 67,
                4: 65,
                5: 62,
                6: 64,
                7: 65,
                8: 65,
                9: 64,
                10: 65,
                11: 67,
                12: 67,
                13: 65,
            },
        }
    )
    mididf = homodf_to_mididf(test_df)
    # So far merely tested by inspection; looks ok.
