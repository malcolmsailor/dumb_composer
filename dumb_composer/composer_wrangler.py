import os
import random
import re
import typing as t
import itertools as it

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.pitch_utils.ranges import Ranger

PREFAB_VOICE_WEIGHTS = {
    "soprano": 0.6,
    "bass": 0.2,
    "tenor": 0.1,
}


class ComposerWrangler:
    _prefab_voices = list(PREFAB_VOICE_WEIGHTS.keys())
    _prefab_weights = list(it.accumulate(PREFAB_VOICE_WEIGHTS.values()))

    def __init__(self):
        self._ranger = Ranger()

    # TODO:
    #   - choose register
    #   - choose disposition (i.e., melody on top, melody in middle, melody
    #       in bass)
    #   - optionally choose time signature?

    def _get_paths(
        self,
        base_path: str,
        exts: t.Set[str] = {"txt", "rntxt"},
        basename_startswith=None,
    ):
        exts = {("" if ext.startswith(".") else ".") + ext for ext in exts}
        for dirpath, _, filenames in os.walk(base_path):
            for f in filenames:
                if basename_startswith is not None and not f.startswith(
                    basename_startswith
                ):
                    continue
                if os.path.splitext(f)[1] in exts:
                    yield os.path.join(dirpath, f)

    def walk_folder(
        self,
        base_path: str,
        exts: t.Set[str] = {"txt", "rntxt"},
        basename_startswith=None,
    ):

        for path in self._get_paths(base_path, exts, basename_startswith):
            self(path)

    def _init_composer_settings(self, prefab_voice):
        if prefab_voice is None:
            prefab_voice = random.choices(
                self._prefab_voices, cum_weights=self._prefab_weights, k=1
            )[0]
        ranges = self._ranger(melody_part=prefab_voice)
        return PrefabComposerSettings(prefab_voice=prefab_voice, **ranges)

    def __call__(self, rntxt_path: str, prefab_voice: t.Optional[str] = None):
        settings = self._init_composer_settings(prefab_voice)
        composer = PrefabComposer(settings)
        out, ts = composer(rntxt_path, return_ts=True)
        return out, ts
