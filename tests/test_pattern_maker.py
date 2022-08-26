from dumb_composer.patterns import PatternMaker
from dumb_composer.shared_classes import Allow


def test_filter_patterns():
    for ts in ("4/4", "3/4", "6/8"):
        pm = PatternMaker(ts)
        patterns = [getattr(pm, pattern) for pattern in pm._patterns]
        if pm.rf.meter.is_compound:
            assert not any(p.allow_compound is Allow.NO for p in patterns)
        else:
            assert not any(p.allow_compound is Allow.ONLY for p in patterns)
        if pm.rf.meter.is_triple:
            assert not any(p.allow_triple is Allow.NO for p in patterns)
        else:
            assert not any(p.allow_triple is Allow.ONLY for p in patterns)
