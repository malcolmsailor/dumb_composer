import random
import os

import pytest

import dumb_composer.two_part_contrapuntist as mod
from midi_to_notes import df_to_midi

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def test_two_part_contrapuntist():

    rn_format = """Time signature: {}
    m1 Bb: I
    m2 F: ii
    m3 I64
    m4 V7
    m5 I
    m6 ii6
    m7 V7
    m8 I
    m9 I64
    m10 V7
    m11 I53
    m12 V43
    m13 V42
    m14 I6
    """
    time_sigs = [(4, 4), (3, 4)]
    for numer, denom in time_sigs:
        tpc = mod.TwoPartContrapuntist()
        random.seed(42)
        ts = f"{numer}/{denom}"
        rn_temp = rn_format.format(ts)
        score = tpc(rn_temp)
        out_df = tpc.get_mididf_from_score(score)
        mid_path = os.path.join(
            TEST_OUT_DIR,
            f"two_part_contrapuntist_ts={ts.replace('/', '-')}.mid",
        )
        print(f"writing {mid_path}")
        df_to_midi(out_df, mid_path, ts=(numer, denom))


@pytest.mark.skip(reason="missing torch import at cabin")
def test_two_part_contrapuntist_from_ml_out():
    tpc = mod.TwoPartContrapuntist()
    test_seqs = """<START> address<1> rn<I> figure<53> tonic<0M> address<0.5> key<7M> rn<ii> figure<53> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<0.5> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<0.5> rn<ii> figure<6> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<0.25> rn<I> figure<53> tonic<0M> address<0.5> rn<I> figure<64> tonic<0M> address<1.25> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<0.5> rn<V> figure<43> tonic<0M> address<1.25> rn<V> figure<42> tonic<0M> address<1> rn<I> figure<6> tonic<0M> address<0.5> rn<V> figure<42> tonic<0M> <PAD>
    <START> address<1> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<1> rn<vi> figure<53> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> address<1> rn<V> figure<7> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<0.5> rn<I> figure<6> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<53> tonic<0M> <PAD>
    <START> address<1> key<2m> rn<V> figure<7> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<0.5> rn<viio> figure<7> tonic<7M> address<1> rn<V> figure<43> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<1> key<0M> rn<V> figure<7> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<1> rn<I> figure<64> tonic<0M> address<1> rn<V> figure<7> tonic<0M> address<1> rn<I> figure<64> tonic<0M> <PAD> <PAD> <PAD> <PAD>
    <START> address<3.0625> rn<V> figure<7> tonic<0m> address<0.5> rn<i> figure<53> tonic<0m> address<1.25> rn<IV> figure<64> tonic<0m> address<1> rn<viio> figure<42> tonic<0m> address<1> rn<v> figure<53> tonic<0m> address<0.25> rn<iv> figure<6> tonic<0m> address<0.5> rn<V> figure<7> tonic<0m> address<1.25> rn<i> figure<64> tonic<0m> address<1> rn<viio> figure<42> tonic<0m> address<0.25> rn<viio> figure<7> tonic<0m> address<0.5> rn<iv> figure<53> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<1> rn<i> figure<53> tonic<0m> address<1> rn<viio> figure<7> tonic<0m> address<0.5> rn<V> figure<7> tonic<0m> <PAD> <PAD>
    <START> address<1> rn<IV> figure<64> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<0.25> key<2m> rn<viio> figure<7> tonic<0m> address<0.5> rn<i> figure<53> tonic<0m> address<1> rn<viio> figure<7> tonic<0m> address<0.25> rn<i> figure<53> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<0.5> rn<i> figure<53> tonic<0m> address<1> rn<V> figure<7> tonic<0m> address<1> key<0M> rn<I> figure<53> tonic<0M> address<1> rn<I> figure<53> tonic<0M> address<0.25> rn<I> figure<6> tonic<0M> address<0.5> rn<V> figure<7> tonic<0M> address<1.25> rn<I> figure<64> tonic<0M> <PAD> <PAD> <PAD> <PAD>"""

    random.seed(42)
    test_seqs = test_seqs.split("\n")
    test_seqs = [seq.strip().split(" ") for seq in test_seqs]
    keys = [random.choice(range(12)) for _ in test_seqs]
    for i, seq in enumerate(test_seqs):
        random.seed(42)
        out_df = tpc.from_ml_out(seq, "4/4", keys[i])
        mid_path = os.path.join(
            TEST_OUT_DIR,
            "two_part_contrapuntist_from_ml_out" + f"_{i + 1}.mid",
        )
        print(f"writing {mid_path}")
        df_to_midi(out_df, mid_path)


if __name__ == "__main__":
    test_two_part_contrapuntist()
