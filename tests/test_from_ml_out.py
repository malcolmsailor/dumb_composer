import inspect
import os
import random

import pytest

from dumb_composer import ml_out_handler
from midi_to_notes import df_to_midi

from tests.test_helpers import TEST_OUT_DIR


def get_funcname():
    return inspect.currentframe().f_back.f_code.co_name


@pytest.mark.skip(reason="missing torch import at cabin")
def test_ml_out_handler():
    funcname = get_funcname()

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
        out_df = ml_out_handler(seq, "4/4", keys[i], text_annotations=True)
        mid_path = os.path.join(
            TEST_OUT_DIR,
            funcname + f"_{i + 1}.mid",
        )
        print(f"writing {mid_path}")
        df_to_midi(out_df, mid_path)
