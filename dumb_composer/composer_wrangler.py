import logging
import os
import random
import re
import typing as t
import itertools as it

from midi_to_notes import df_to_midi
from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.pitch_utils.ranges import Ranger
from dumb_composer.time import MeterError

PREFAB_VOICE_WEIGHTS = {
    "soprano": 0.7,
    "bass": 0.15,
    "tenor": 0.15,
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

    def _init_composer_settings(self, prefab_voice, transpose):
        if prefab_voice is None:
            prefab_voice = random.choices(
                self._prefab_voices, cum_weights=self._prefab_weights, k=1
            )[0]
            logging.debug(f"setting prefab_voice to '{prefab_voice}'")
        if transpose is None:
            transpose = random.randrange(12)
            logging.debug(f"setting transpose to {transpose}")
        ranges = self._ranger(melody_part=prefab_voice)
        return (
            PrefabComposerSettings(prefab_voice=prefab_voice, **ranges),
            transpose,
        )

    def __call__(
        self,
        rntxt_path: str,
        prefab_voice: t.Optional[str] = None,
        transpose: t.Optional[int] = None,
    ):
        logging.info(f"Input path: {rntxt_path}")
        settings, transpose = self._init_composer_settings(
            prefab_voice, transpose
        )
        composer = PrefabComposer(settings)
        out, ts = composer(rntxt_path, return_ts=True, transpose=transpose)
        return out, ts

    @staticmethod
    def _default_path_formatter(path, i, transpose, prefab_voice):
        return (
            os.path.splitext(os.path.basename(path))[0]
            + f"_{prefab_voice}_transpose={transpose}_{i+1:03d}"
        )

    @staticmethod
    def _change_logger(logpath):
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        log.addHandler(logging.FileHandler(logpath, "w"))

    def call_n_times(
        self,
        n: int,
        output_dir,
        paths: t.Sequence[str],
        shuffle: bool = True,
        random_transpose: bool = True,
        path_formatter: t.Optional[t.Callable[[str, int, int], str]] = None,
        _pytestconfig=None,
        _log_wo_pytest=False,
    ):
        paths_todo = []
        if path_formatter is None:
            path_formatter = self._default_path_formatter
        missing_files = n - len(paths_todo)
        while missing_files:
            if shuffle:
                paths_todo.extend(
                    random.sample(paths, min(missing_files, len(paths)))
                )
            else:
                paths_todo.extend(paths[: min(missing_files, len(paths))])
            missing_files = n - len(paths_todo)
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__} making {n} files")
        if _log_wo_pytest:
            logging.basicConfig(filename="log_file_path", level="DEBUG")
        errors = []
        skipped = []

        for i, path in enumerate(paths_todo):
            print(f"{i + 1}/{len(paths_todo)}: {path}")
            if random_transpose:
                transpose = random.choice(range(12))
            else:
                transpose = 0
            prefab_voice = random.choices(
                self._prefab_voices, cum_weights=self._prefab_weights, k=1
            )[0]
            output_path_wo_ext = os.path.join(
                output_dir, path_formatter(path, i, transpose, prefab_voice)
            )
            mid_path = f"{output_path_wo_ext}.mid"
            log_path = f"{output_path_wo_ext}.log"
            if _pytestconfig is not None:
                logging_plugin = _pytestconfig.pluginmanager.get_plugin(
                    "logging-plugin"
                )
                logging_plugin.set_log_path(log_path)
            elif _log_wo_pytest:
                self._change_logger(log_path)
            try:
                out, ts = self(path, prefab_voice=prefab_voice)
            except KeyboardInterrupt:
                raise
            except MeterError as exc:
                print(f"Skipping due to meter error: {exc}")
                skipped.append(path)
            except:
                print(f"ERROR: {path}")
                errors.append(path)
            df_to_midi(out, mid_path, ts)
        if skipped:
            print(f"{len(skipped)} files skipped:")
            for path in skipped:
                print(path)
        if errors:
            print(f"{len(errors)} errors:")
            for path in errors:
                print(path)
        print(f"{len(skipped)} files skipped. {len(errors)} files had errors.")
