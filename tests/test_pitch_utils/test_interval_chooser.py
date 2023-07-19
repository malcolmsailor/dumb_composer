import os
import sys

import pytest

import dumb_composer.pitch_utils.interval_chooser as mod
from tests.test_utils.shell_plot import print_histogram


@pytest.mark.parametrize("custom_weights", (None, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
def test_interval_chooser(custom_weights):
    intervals = [-3, -2, -1, 0, 1, 2, 3]
    n = 10000
    for smaller_mel_interval_concentration in (0, 0.5, 1.0, 2.0):
        for unison_weighted_as in (0, 1, 3):
            icsettings = mod.IntervalChooserSettings(
                smaller_mel_interval_concentration=smaller_mel_interval_concentration,
                unison_weighted_as=unison_weighted_as,
            )
            ic = mod.IntervalChooser(icsettings)
            result = ic.choose_intervals(
                intervals, n * 10, custom_weights=custom_weights
            )
            print_histogram(
                result,
                bins=intervals + [max(intervals) + 1],
                name=f"smaller_mel_interval_concentration={smaller_mel_interval_concentration}, unison_weighted_as={unison_weighted_as}",
            )


@pytest.mark.parametrize("octave", (-12, 0, 12))
@pytest.mark.parametrize("custom_weights", (None,))
@pytest.mark.parametrize("other_pitch", (0, 4))
def test_harmonically_informed_interval_chooser(octave, custom_weights, other_pitch):
    melodic_intervals = [x + octave for x in range(13)]
    harmonic_intervals = [
        melodic_interval - other_pitch for melodic_interval in melodic_intervals
    ]
    n = 10000
    ic = mod.HarmonicallyInformedIntervalChooser()
    result = ic.choose_intervals(
        melodic_intervals, harmonic_intervals, n=n, custom_weights=custom_weights
    )
    print_histogram(result, bins=melodic_intervals + [max(melodic_intervals) + 1])
