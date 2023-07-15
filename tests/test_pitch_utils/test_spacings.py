import pytest

from dumb_composer.pitch_utils.spacings import (
    RangeConstraints,
    SpacingConstraints,
    _yield_spacing_helper,
    yield_spacings,
)


@pytest.mark.parametrize(
    "pcs", ((0, 0, 4, 4), (0, 0, 4, 7), (0, 0, 0, 4), (0, 4, 4, 7), (0, 4, 7, 7))
)
@pytest.mark.parametrize("melody_pitch", (None, 72))
@pytest.mark.parametrize("shuffled", (True, False))
def test_yield_spacing_helper(pcs, melody_pitch, shuffled):
    result = list(
        _yield_spacing_helper(
            (),
            remaining_pcs=list(pcs),
            range_constraints=RangeConstraints(),
            spacing_constraints=SpacingConstraints(),
            melody_pitch=melody_pitch,
            shuffled=shuffled,
        )
    )
    assert len(set(result)) == len(result)
    if melody_pitch:
        assert all(tuple(sorted(x % 12 for x in item[:-1])) == pcs for item in result)
    else:
        assert all(tuple(sorted(x % 12 for x in item)) == pcs for item in result)


@pytest.mark.parametrize(
    "pcs", ((0, 0, 4, 4), (0, 0, 4, 7), (0, 0, 0, 4), (0, 4, 4, 7), (0, 4, 7, 7))
)
@pytest.mark.parametrize("shuffled", (True, False))
def test_yield_spacings(pcs, shuffled):
    result = list(yield_spacings(pcs, shuffled=shuffled))
    assert len(set(result)) == len(result)
