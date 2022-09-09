import music21

import dumb_composer.pitch_utils.music21_handler


def test_rn_transpose():
    for transpose in range(1, 13):
        rn = music21.roman.RomanNumeral("I", "C")
        rn2 = rn.transpose(transpose)
        assert rn2.pitchClasses == [
            (pc + transpose) % 12 for pc in rn.pitchClasses
        ]
        assert (
            rn2.key.tonic.pitchClass
            == (rn.key.tonic.pitchClass + transpose) % 12
        )
