import os
import random
from collections import Counter, defaultdict
from typing import Iterable

import dumb_composer.two_part_contrapuntist as mod
import pandas as pd
import pytest
from dumb_composer.pitch_utils.intervals import reduce_compound_interval
from dumb_composer.utils.recursion import DeadEnd
from dumb_composer.utils.shell_plot import print_bar
from midi_to_note_table import df_to_midi

from tests.test_helpers import TEST_OUT_DIR


@pytest.mark.parametrize(
    "rntxt, structural_soprano_pitches",
    (
        ("m1 d: i b3 viio7", [62, 65, 69]),
        ("m1 F: V7 b3 viio7/vi", [60, 64, 67, 70]),
        ("m1 F: V7 b3 #viio7/vi", [60, 64, 67, 70]),
    ),
)
@pytest.mark.parametrize("do_first", (mod.OuterVoice.BASS, mod.OuterVoice.MELODY))
def test_avoid_doubling_tendency_tones(rntxt, structural_soprano_pitches, do_first):
    random.seed(42)
    settings = mod.TwoPartContrapuntistSettings(do_first=do_first)
    for melody_pitch in structural_soprano_pitches:
        tpc = mod.TwoPartContrapuntist(chord_data=rntxt, settings=settings)
        # We hack the score so that it starts with the intended pitch
        tpc._score._score.structural_bass.append(tpc._score._score.chords[0].foot + 24)
        tpc._score._score.structural_soprano.append(melody_pitch)
        try:
            for pitches in tpc.step():
                assert pitches["melody"] % 12 != pitches["bass"] % 12
        except DeadEnd:
            pass


def update_counts(score, counts: defaultdict[str, Counter]):
    paired_list = list(zip(score.structural_bass, score.structural_soprano))
    for (prev_bass, prev_mel), (bass, mel) in zip(
        [(None, None)] + paired_list[:-1], paired_list
    ):
        if prev_bass is not None:
            bass_mel_interval = bass - prev_bass
            counts["bass_mel_interval"][bass_mel_interval] += 1
        if prev_mel is not None:
            melody_mel_interval = mel - prev_mel
            counts["melody_mel_interval"][melody_mel_interval] += 1
        counts["reduced_harmonic_interval"][reduce_compound_interval(mel - bass)] += 1


@pytest.mark.parametrize("time_sig", [(4, 4), (3, 4)])
@pytest.mark.parametrize("do_first", (mod.OuterVoice.MELODY, mod.OuterVoice.BASS))
def test_two_part_contrapuntist(time_sig, do_first):
    numer, denom = time_sig
    ts = f"{numer}/{denom}"
    rn_txt = """Time signature: {ts}
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
    m15 viio6
    m16 I
    m17 V6
    m18 viio6/V
    m18 V
    """

    dfs = []

    settings = mod.TwoPartContrapuntistSettings(do_first=do_first)
    counts = defaultdict(Counter)

    for seed in range(42, 42 + 10):
        tpc = mod.TwoPartContrapuntist(chord_data=rn_txt, settings=settings)
        random.seed(seed)
        score = tpc()
        update_counts(score._score, counts)
        out_df = tpc.get_mididf_from_score(score._score)
        dfs.append(out_df)

    time_adjustment = 0
    for df in dfs:
        df["onset"] += time_adjustment
        df["release"] += time_adjustment
        time_adjustment = df["release"].max() + numer
    do_first_str = "bass" if do_first is mod.OuterVoice.BASS else "melody"
    mid_path = os.path.join(
        TEST_OUT_DIR,
        f"two_part_contrapuntist_ts={ts.replace('/', '-')}_do-first={do_first_str}.mid",
    )
    out_df = pd.concat(dfs, axis=0)
    print(f"writing {mid_path}")

    for title, counter in counts.items():
        print(counter)
        print_bar(
            name=f"{title}: {ts=} {do_first=}",
            counter=counter,
            horizontal=True,
            sort_by_key=True,
            char_height=65,
        )
    df_to_midi(out_df, mid_path, ts=(numer, denom))


@pytest.mark.skip(reason="missing torch import at cabin")
def test_two_part_contrapuntist_from_ml_out():
    tpc = mod.TwoPartContrapuntist()  # type:ignore
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
