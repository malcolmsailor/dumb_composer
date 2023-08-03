import os
import random

import pandas as pd
import pytest
from midi_to_notes import df_to_midi

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.time import MeterError
from dumb_composer.utils.recursion import RecursionFailed
from tests.test_helpers import TEST_OUT_DIR, get_funcname, merge_dfs, write_df


@pytest.mark.parametrize(
    "time_sig",
    [
        (4, 4),
        # (3, 4),
    ],
)
# TODO: (Malcolm 2023-07-28) add tenor and maybe alto
@pytest.mark.parametrize(
    "prefab_voices",
    (
        # ("soprano",),
        # ("bass",),
        # ("soprano", "alto", "tenor", "bass"),
        ("soprano", "bass"),
    ),
)
def test_prefab_composer(quick, pytestconfig, time_sig, prefab_voices):
    numer, denom = time_sig
    ts = f"{numer}/{denom}"
    rn_txt = f"""Time signature: {ts}
    m1 Bb: I
    m2 V7/IV
    m3 IV64
    Note: TODO try a pedal point here
    m4 V65
    m5 I b3 V43
    m6 I6 b3 I
    m7 F: viio64 b3 viio6/ii
    m8 ii b3 ii42
    m9 V65 b3 V7
    m10 vi b3 viio7/V
    m11 V b3 Cad64
    m12 V b3 V7
    m13 I
    """
    funcname = get_funcname()
    test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
    os.makedirs(test_out_dir, exist_ok=True)

    path_wo_ext = os.path.join(
        test_out_dir,
        f"ts={ts.replace('/', '-')}_" f"prefab_voice={prefab_voices}",
    )
    mid_path = path_wo_ext + ".mid"
    log_path = path_wo_ext + ".log"
    logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
    logging_plugin.set_log_path(log_path)

    dfs = []
    initial_seed = 48
    number_of_tries = 10
    for seed in range(initial_seed, initial_seed + number_of_tries):
        random.seed(seed)
        settings = PrefabComposerSettings(
            prefab_voices=prefab_voices,
            top_down_tie_prob={0: 0.5, 1: 1.0},  # TODO move elsewhere
        )
        pfc = PrefabComposer(settings)
        out_df = pfc(rn_txt)
        dfs.append(out_df)

    out_df = merge_dfs(dfs, ts)

    write_df(
        out_df,
        mid_path,
        ts=(numer, denom),
    )
    print(f"log_path = {log_path}")

    if quick:
        raise NotImplementedError()


def test_problem_files(slow):
    if not slow:
        return
    problem_files = [
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Early_Choral_Bach,_Johann_Sebastian_Chorales_07.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Etudes_and_Preludes_Bach,_Johann_Sebastian_The_Well-Tempered_Clavier_I_05.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Etudes_and_Preludes_Bach,_Johann_Sebastian_The_Well-Tempered_Clavier_I_12.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Etudes_and_Preludes_Bach,_Johann_Sebastian_The_Well-Tempered_Clavier_I_13.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Chaminade,_Cécile_Amoroso.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Chausson,_Ernest_7_Mélodies,_Op.2_7_Le_Colibri.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Coleridge-Taylor,_Samuel_6_Sorrow_Songs,_Op.57_6_Too_late_for_love.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Hensel,_Fanny_(Mendelssohn)_5_Lieder,_Op.10_5_Bergeslust.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Schubert,_Franz_Schwanengesang,_D.957_01_Liebesbotschaft.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Schubert,_Franz_Winterreise,_D.911_17_Im_Dorfe.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Schubert,_Franz_Winterreise,_D.911_21_Das_Wirthshaus.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Schumann,_Robert_Dichterliebe,_Op.48_08_Und_wüssten’s_die_Blumen.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Schumann,_Robert_Dichterliebe,_Op.48_11_Ein_Jüngling_liebt_ein_Mädchen.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Wolf,_Hugo_Eichendorff-Lieder_13_Der_Scholar.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/OpenScore-LiederCorpus_Wolf,_Hugo_Eichendorff-Lieder_20_Waldmädchen.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No2_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No3_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No3_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No4_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No5_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No6_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No1_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No2_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No2_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No3_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No3_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op074_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op074_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op130_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op130_6.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op131_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op131_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_2.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_5.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op135_1.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Haydn,_Franz_Joseph_Op20_No1_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Haydn,_Franz_Joseph_Op20_No2_3.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Haydn,_Franz_Joseph_Op20_No6_4.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Bach,_Johann_Sebastian_B_Minor_mass,_BWV232_Crucifixus.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_65_A.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_66_A.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_71_B.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_75_B.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Mozart,_Wolfgang_Amadeus_K613_B.txt",
        "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Purcell,_Henry_Sonata_Z807.txt",
    ]
    for i, f in enumerate(problem_files):
        print(f"{i + 1}/{len(problem_files)}: {f}")
        if f in (
            # files where fixes to music21 are pending
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No3_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No4_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No5_3.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op018_No6_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No1_2.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No2_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No2_3.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No3_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op059_No3_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op074_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op074_3.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_2.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op095_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_2.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op127_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op130_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op130_6.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op131_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op131_4.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_1.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_2.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_3.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op132_5.txt",
            "/Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Quartets_Beethoven,_Ludwig_van_Op135_1.txt",
        ):
            continue
        composer = PrefabComposer()
        try:
            composer(f)
        except MeterError:
            pass
        except RecursionError:
            print("Recursion error")
            pass


"""
TODO Several of the "problem files" from the Mozart/Beethoven variations have
Recursion errors. Is this because the files are simply too long? Worth 
investigating. Perhaps I can restart the recursive stack every certain number
of chords so that it doesn't get arbitrarily deep. (If it's made it 100 chords,
it's unlikely to have to go back to the first chord in any case.)
"""
"""
48/53: /Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_65_A.txt
Reading score... done.
Recursion error
49/53: /Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_66_A.txt
Reading score... done.
Recursion error
50/53: /Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_71_B.txt
Reading score... done.
Recursion error
51/53: /Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Beethoven,_Ludwig_van_WoO_75_B.txt
Reading score... done.
Recursion error
52/53: /Users/malcolm/datasets/When-in-Rome/Corpus/../analyses_only/Variations_and_Grounds_Mozart,_Wolfgang_Amadeus_K613_B.txt
Reading score... done.
Recursion error"""
