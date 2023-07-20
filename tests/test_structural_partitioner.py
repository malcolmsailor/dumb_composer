from collections import Counter

from dumb_composer.pitch_utils.chords import get_chords_from_rntxt
from dumb_composer.shared_classes import Score
from dumb_composer.structural_partitioner import StructuralPartitioner


def count_chord_durations(score):
    return Counter([chord.release - chord.onset for chord in score.chords])


def test_structural_partitioner(time_sig=(4, 4)):
    numer, denom = time_sig
    ts = f"{numer}/{denom}"
    rntxt = f"""Time signature: {ts}
    m1 C: I
    m2 V
    m3 I b3 V
    m4 I
    m6 V
    m8 I b2 V b3 I b4 V
    m9 b3 I
    m10 b2 V
    m11 I
    """
    chord_data = get_chords_from_rntxt(rntxt)  # type:ignore
    for _ in range(10):
        score = Score(chord_data, ts=ts)
        print("")
        print("Original duration counts:")
        print(count_chord_durations(score))
        structural_partitioner = StructuralPartitioner()
        structural_partitioner(score)
        print("Modified duration counts:")
        print(count_chord_durations(score))
