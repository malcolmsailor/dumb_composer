import os
import random

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.time import MeterError
from dumb_composer.utils.recursion import RecursionFailed
from tests.test_helpers import TEST_OUT_DIR, get_funcname, write_df


def test_prefab_composer(quick, pytestconfig):
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
    funcname = get_funcname()
    test_out_dir = os.path.join(TEST_OUT_DIR, funcname)
    os.makedirs(test_out_dir, exist_ok=True)
    time_sigs = [(4, 4), (3, 4)]
    for numer, denom in time_sigs:
        for prefab_voice in ("soprano", "tenor", "bass"):
            random.seed(42)
            for i in range(1):
                ts = f"{numer}/{denom}"
                path_wo_ext = os.path.join(
                    test_out_dir,
                    f"ts={ts.replace('/', '-')}_"
                    f"prefab_voice={prefab_voice}_{i + 1}",
                )
                mid_path = path_wo_ext + ".mid"
                log_path = path_wo_ext + ".log"
                logging_plugin = pytestconfig.pluginmanager.get_plugin("logging-plugin")
                logging_plugin.set_log_path(log_path)
                rn_temp = rn_format.format(ts)
                settings = PrefabComposerSettings(
                    prefab_voice=prefab_voice,
                    top_down_tie_prob={0: 0.5, 1: 1.0},  # TODO move elsewhere
                )
                pfc = PrefabComposer(settings)
                out_df = pfc(rn_temp)
                write_df(
                    out_df,
                    mid_path,
                    ts=(numer, denom),
                )
                if quick:
                    return


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
