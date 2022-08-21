import random
import os

from dumb_composer.utils.df_helpers import merge_note_dfs
from dumb_composer.dumb_accompanist import DumbAccompanist

from dumb_composer.two_part_contrapuntist import TwoPartContrapuntist
from dumb_composer.prefab_applier import PrefabApplier
from midi_to_notes import df_to_midi

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def test_prefab_applier():
    tpc = TwoPartContrapuntist()
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
    time_sigs = [(4, 4)]
    for numer, denom in time_sigs:
        random.seed(42)
        ts = f"{numer}/{denom}"
        rn_temp = rn_format.format(ts)
        score = tpc(rn_temp)
        pa = PrefabApplier()
        out_df = pa(score)
        # mid_path = os.path.join(
        #     TEST_OUT_DIR,
        #     f"prefab_applier={ts.replace('/', '-')}.mid",
        # )
        # print(f"writing {mid_path}")
        # df_to_midi(out_df, mid_path, ts=(numer, denom))
        # dc = DumbAccompanist()
        # dc_out_df = dc(rn_temp)
        # dc_out_df["track"] = 2
        # merged_df = merge_note_dfs(out_df[out_df.track == 1], dc_out_df)
        # mid_path = os.path.join(
        #     TEST_OUT_DIR,
        #     f"prefab_applier_merged_with_dumb_composer={ts.replace('/', '-')}.mid",
        # )
        # print(f"writing {mid_path}")
        # df_to_midi(merged_df, mid_path, ts=(numer, denom))
