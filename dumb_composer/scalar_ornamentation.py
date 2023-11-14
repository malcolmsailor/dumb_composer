"""
Eventually I'd like to include arpeggio ornamentation as well.
"""

from dumb_composer.chords.chords import Chord
from dumb_composer.pitch_utils.types import Pitch


def get_approach_tones(
    dst_pitch: Pitch, src_chord: Chord, dst_chord: Chord
) -> tuple[Pitch, ...]:
    """
    # >>> rntxt = '''m1 C: I V'''
    # >>> I, V = get_chords_from_rntxt(rntxt)
    # >>> get_approach_tones(62, src_chord=I, dst_chord=V)
    # (60, 64)
    """
    raise NotImplementedError
    pass
