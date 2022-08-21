import dumb_composer.pitch_utils.intervals as mod


def test_interval_finder():
    result = mod.interval_finder(62, (0, 4, 7), 48, 72)
