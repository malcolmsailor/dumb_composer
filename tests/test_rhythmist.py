import random
import os
import dumb_composer.rhythmist as mod


def test_rhythmist():
    rhythmist = mod.DumbRhythmist(4.0)
    r = rhythmist(4)
