import dumb_composer.pitch_utils.interval_chooser as mod

import os
import sys

sys.path.append(
    os.path.join(os.path.dirname((os.path.realpath(__file__))), "../test_utils")
)

from shell_plot import print_histogram  # type:ignore


def test_interval_chooser():

    intervals = [-3, -2, -1, 0, 1, 2, 3]
    n = 10000
    for lambda_ in (0, 0.5, 1.0, 2.0):
        for unison_weighted_as in (0, 1, 3):
            icsettings = mod.IntervalChooserSettings(
                lambda_=lambda_, unison_weighted_as=unison_weighted_as
            )
            ic = mod.IntervalChooser(icsettings)
            result = ic.choose_intervals(intervals, n * 10)
            print_histogram(
                result,
                bins=intervals + [max(intervals) + 1],
                name=f"lambda_={lambda_}, unison_weighted_as={unison_weighted_as}",
            )
