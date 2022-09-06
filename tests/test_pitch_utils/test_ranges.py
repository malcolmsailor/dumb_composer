from collections import defaultdict
from dumb_composer.pitch_utils.ranges import Ranger

N_TRIALS = 1000


def test_ranger():
    ranger = Ranger(slope_scale=1.0)
    ambituses = defaultdict(lambda: defaultdict(list))
    for _ in range(N_TRIALS):
        out = ranger(melody_part="soprano")
        soprano_range = out["mel_range"]
        bass_range = out["bass_range"]
        total_ambitus = soprano_range[1] - bass_range[0]
        ambituses["soprano"][total_ambitus].append(
            soprano_range[1] - soprano_range[0]
        )

        ambituses["bass"][total_ambitus].append(bass_range[1] - bass_range[0])
    mean_soprano_ambituses = {
        x: sum(ambituses["soprano"][x]) / len(ambituses["soprano"][x])
        for x in sorted(ambituses["soprano"])
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
