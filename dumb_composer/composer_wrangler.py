import pytest

pytest.mark.skip(reason="need to update implementation")

import ast
import itertools as it
import json
import logging
import os
import random
import re
import typing as t
from dataclasses import asdict

from midi_to_notes import df_to_midi

from dumb_composer.dumb_composer import PrefabComposer, PrefabComposerSettings
from dumb_composer.pitch_utils.music21_handler import transpose_and_write_rntxt
from dumb_composer.pitch_utils.ranges import Ranger
from dumb_composer.time import MeterError

PREFAB_VOICE_WEIGHTS = {
    "soprano": 0.7,
    "bass": 0.15,
    "tenor": 0.15,
}


def read_settings(path: str):
    with open(path, "r") as inf:
        settings_dict = ast.literal_eval(inf.read())
    return settings_dict


class ComposerWrangler:
    _prefab_voices = list(PREFAB_VOICE_WEIGHTS.keys())
    _prefab_weights = list(it.accumulate(PREFAB_VOICE_WEIGHTS.values()))

    def __init__(self):
        self._ranger = Ranger()

    # TODO?
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

    def _init_composer_settings(self, settings_dict, prefab_voice, transpose):
        if settings_dict is None:
            settings_dict = {}
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
            PrefabComposerSettings(
                prefab_voice=prefab_voice, **ranges, **settings_dict
            ),
            transpose,
        )

    def __call__(
        self,
        rntxt_path: str,
        settings_dict: t.Optional[dict] = None,
        prefab_voice: t.Optional[str] = None,
        transpose: t.Optional[int] = None,
    ):
        logging.info(f"Input path: {rntxt_path}")
        settings, transpose = self._init_composer_settings(
            settings_dict, prefab_voice, transpose
        )
        composer = PrefabComposer(settings)
        out = composer(rntxt_path, return_ts=True, transpose=transpose)
        return out

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
        write_midi: bool = True,
        write_csv: bool = False,
        write_romantext: bool = False,
        settings_path: t.Optional[str] = None,
        _pytestconfig=None,
        _log_wo_pytest=False,
    ):
        def _log_composer_settings(settings_dict):
            generic_settings = PrefabComposerSettings(
                prefab_voices="varies",
                # mel_range="varies",
                # bass_range="varies",
                # accomp_range="varies",
                **settings_dict,
            )
            with open(os.path.join(output_dir, "settings.txt"), "w") as outf:
                json.dump(asdict(generic_settings), outf, indent=2, default=repr)

        if not any((write_midi, write_csv, write_romantext)):
            raise ValueError(
                "all of `write_midi`, `write_csv`, and `write_romantext` are "
                "False; there is nothing to do"
            )
        paths_todo = []
        if path_formatter is None:
            path_formatter = self._default_path_formatter
        missing_files = n - len(paths_todo)
        while missing_files:
            if shuffle:
                paths_todo.extend(random.sample(paths, min(missing_files, len(paths))))
            else:
                paths_todo.extend(paths[: min(missing_files, len(paths))])
            missing_files = n - len(paths_todo)
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__} making {n} files")
        if _log_wo_pytest:
            logging.basicConfig(filename="log_file_path", level="DEBUG")
        errors = []
        skipped = []
        if settings_path is None:
            settings_dict = {}
        else:
            settings_dict = read_settings(settings_path)
        _log_composer_settings(settings_dict)

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

            log_path = f"{output_path_wo_ext}.log"

            if _pytestconfig is not None:
                logging_plugin = _pytestconfig.pluginmanager.get_plugin(
                    "logging-plugin"
                )
                logging_plugin.set_log_path(log_path)
            elif _log_wo_pytest:
                self._change_logger(log_path)
            try:
                out, ts = self(
                    path,
                    settings_dict,
                    prefab_voice=prefab_voice,
                    transpose=transpose,
                )
            except KeyboardInterrupt:
                raise
            except MeterError as exc:
                print(f"Skipping due to meter error: {exc}")
                skipped.append(path)
            except Exception as exc:
                print(f"ERROR: {path}")
                errors.append(path)
            else:
                if write_midi:
                    mid_path = f"{output_path_wo_ext}.mid"
                    df_to_midi(out, mid_path, ts)
                if write_csv:
                    csv_path = f"{output_path_wo_ext}.csv"
                    out.to_csv(csv_path)
                if write_romantext:
                    new_rntxt_path = f"{output_path_wo_ext}.rntxt"
                    contents = transpose_and_write_rntxt(path, transpose)
                    with open(new_rntxt_path, "w") as outf:
                        outf.write(contents)

        if skipped:
            print(f"{len(skipped)} files skipped:")
            for path in skipped:
                print(path)
        if errors:
            print(f"{len(errors)} errors:")
            for path in errors:
                print(path)
        print(f"Output folder: {output_dir}")
        print(f"{len(skipped)} files skipped. {len(errors)} files had errors.")
