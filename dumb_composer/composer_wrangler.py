import os
import re
import typing as t

from dumb_composer.dumb_composer import PrefabComposer


class ComposerWrangler:
    def __init__(self):
        pass

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

    def __call__(self, rntxt_path: str):
        composer = PrefabComposer()
        out, ts = composer(rntxt_path, return_ts=True)
        return out, ts
