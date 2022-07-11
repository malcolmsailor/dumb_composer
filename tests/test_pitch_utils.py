import dumb_composer.pitch_utils as pu


def test_put_in_range():
    for tet in (12, 31):
        range_ = (60, 60 + tet)
        low, high = range_
        for p in range(24, 96):
            assert low <= pu.put_in_range(p, low, high, tet=tet) <= high
            assert low <= pu.put_in_range(p, low=low, tet=tet)
            assert pu.put_in_range(p, high=high, tet=tet) <= high
        range_ = (60, 61)
        low, high = range_
        for p in range(24, 96):
            p2 = pu.put_in_range(p, low, high, tet=tet)
            assert (low <= p2 <= low + tet) or (high - tet <= p2 <= high)
