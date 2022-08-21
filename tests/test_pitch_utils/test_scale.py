from dumb_composer.pitch_utils.scale import Scale

import pytest


def test_scale2():
    scale = Scale([7, 9, 11, 0, 2, 4, 6])  # G major
    assert scale.tonic_pc == 7
    pentatonic_scale = Scale([5, 7, 9, 0, 2])  # F pentatonic
    assert pentatonic_scale.nearest_index(65) == 25  # F5, returns 5 * 5 for F5


def test_scale():
    for bad_pcs in ((4, 4, 6, 8), (4, 2, 6, 8), (-2, 0, 4), (0, 4, 12), ()):
        with pytest.raises(AssertionError):
            Scale(bad_pcs)

    pcs_list = [
        (0, 2, 4, 5, 7, 9, 11),
        (0, 4, 8),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ]
    for pcs in pcs_list:
        for tonic_idx in (0, 1, 2):
            rotated_pcs = pcs[:tonic_idx] + pcs[tonic_idx:]
            for zero_pitch in (-12, 0, 14):
                scale = Scale(rotated_pcs, zero_pitch=zero_pitch)
                for sd in range(-20, 20):
                    pitch = scale[sd]
                    assert scale.index(pitch) == sd
