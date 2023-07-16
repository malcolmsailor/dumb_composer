import os
import random

import pytest
from midi_to_notes import df_to_midi

import dumb_composer.two_part_contrapuntist as mod
from dumb_composer.shared_classes import PrefabScore
from dumb_composer.utils.recursion import DeadEnd
from tests.test_helpers import TEST_OUT_DIR


@pytest.mark.parametrize(
    "rntxt, structural_melody_pitches",
    (
        ("m1 d: i b3 viio7", [62, 65, 69]),
        ("m1 F: V7 b3 viio7/vi", [60, 64, 67, 70]),
        ("m1 F: V7 b3 #viio7/vi", [60, 64, 67, 70]),
    ),
)
@pytest.mark.parametrize("do_first", (mod.OuterVoice.MELODY,))
def test_avoid_doubling_tendency_tones(rntxt, structural_melody_pitches, do_first):
    random.seed(42)
    settings = mod.TwoPartContrapuntistSettings(do_first=do_first)
    tpc = mod.TwoPartContrapuntist(settings)
    score = PrefabScore(chord_data=rntxt)
    print(f"{rntxt=} {structural_melody_pitches=}")
    for melody_pitch in structural_melody_pitches:
        print(f"first{melody_pitch=}")
        score.structural_bass.clear()
        score.structural_bass.append(score.chords[0].foot + 24)
        score.structural_melody.clear()
        score.structural_melody.append(melody_pitch)
        print(f"{score.structural_bass=} {score.structural_melody=}")
        try:
            for pitches in tpc._step(score):
                assert pitches["melody"] % 12 != pitches["bass"] % 12
                print(f"{pitches=}")
        except DeadEnd:
            pass


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
