import random
import os

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.utils.recursion import RecursionFailed
from test_helpers import write_df
from tests.test_helpers import get_funcname, TEST_OUT_DIR


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
                logging_plugin = pytestconfig.pluginmanager.get_plugin(
                    "logging-plugin"
                )
                logging_plugin.set_log_path(log_path)
                rn_temp = rn_format.format(ts)
                settings = PrefabComposerSettings(prefab_voice=prefab_voice)
                pfc = PrefabComposer(settings)
                out_df = pfc(rn_temp)
                write_df(
                    out_df,
                    mid_path,
                    ts=(numer, denom),
                )
                if quick:
                    return
