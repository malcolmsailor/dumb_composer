import dumb_composer.pitch_utils.put_in_range as pu


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
            p2 = pu.put_in_range(p, low, high, tet=tet, fail_silently=True)
            assert (low <= p2 <= low + tet) or (high - tet <= p2 <= high)


def test_get_all_in_range():
    result = pu.get_all_in_range((1, 4, 7, 10), 0, 0)
    result = pu.get_all_in_range((0, 4, 7), 48, 72)
    assert min(result) == 48
    assert max(result) == 72
    for min_pitch in (0, 48, 72):
        for max_pitch in range(min_pitch, min_pitch + 24):
            for chord in ((0,), (), (0, 4, 7), (1, 4, 7, 10)):
                result = pu.get_all_in_range(chord, min_pitch, max_pitch)
                if len(result) == 0:
                    continue
                assert min(result) >= min_pitch
                assert max(result) <= max_pitch
