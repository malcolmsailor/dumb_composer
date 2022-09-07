from collections import defaultdict
from dumb_composer.pitch_utils.ranges import Ranger

N_TRIALS = 1000


def test_ranger():
    ranger = Ranger(slope_scale=1.0)
    for voice in ("soprano", "bass", "tenor"):
        ambituses = defaultdict(lambda: defaultdict(list))
        for _ in range(N_TRIALS):
            out = ranger(melody_part=voice)
            melody_range = out["mel_range"]
            bass_range = out["bass_range"]
            total_ambitus = max(x[1] for x in out.values() if x[1] != 0) - min(
                x[0] for x in out.values() if x[0] != 0
            )
            ambituses["melody"][total_ambitus].append(
                melody_range[1] - melody_range[0]
            )

            ambituses["bass"][total_ambitus].append(
                bass_range[1] - bass_range[0]
            )
        mean_melody_ambituses = {
            x: sum(ambituses["melody"][x]) / len(ambituses["melody"][x])
            for x in sorted(ambituses["melody"])
        }
        mean_bass_ambituses = {
            x: sum(ambituses["bass"][x]) / len(ambituses["bass"][x])
            for x in sorted(ambituses["bass"])
        }
        # TODO complete this test. The ambituses should tend to get bigger
        #   as the total ambitus gets bigger. On inspection, this is indeed
        #   the case with the following caveats:
        #       - even and odd ambituses, respectively, are sorted, but
        #           they are mis-sorted when combined. Maybe this has to do with
        #           Python's rounding behavior?
        #       - the sorting is probabilistic, so doesn't always work out perfectly
