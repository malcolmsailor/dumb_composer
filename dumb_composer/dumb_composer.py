from collections import Counter
import logging
import typing as t
from dataclasses import dataclass
import itertools as it

import pandas as pd

from .chord_spacer import NoSpacings

from .structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)

from .dumb_accompanist import (
    AccompAnnots,
    DumbAccompanist,
    DumbAccompanistSettings,
)

from .shared_classes import Note, Score
from dumb_composer.pitch_utils.chords import Chord

from .utils.recursion import DeadEnd, RecursionFailed, append_attempt

from dumb_composer.pitch_utils.chords import get_chords_from_rntxt
from dumb_composer.pitch_utils.scale import ScaleDict
from dumb_composer.prefabs.prefab_pitches import MissingPrefabError
from dumb_composer.two_part_contrapuntist import (
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.prefab_applier import PrefabApplier, PrefabApplierSettings


@dataclass
class PrefabComposerSettings(
    TwoPartContrapuntistSettings,
    DumbAccompanistSettings,
    PrefabApplierSettings,
    StructuralPartitionerSettings,
):
    max_recurse_calls: int = 1000
    print_missing_prefabs: bool = True

    def __post_init__(self):
        # We need to reconcile DumbAccompanistSettings'
        # settings with PrefabApplierSettings 'prefab_voice' setting.
        if self.prefab_voice == "bass":
            self.accompaniment_below = None
            self.accompaniment_above = ["prefabs"]
            self.include_bass = False
        elif self.prefab_voice == "tenor":
            # TODO set ranges in a better/more dynamic way
            self.mel_range = (48, 67)
            self.accomp_range = (60, 84)
            self.accompaniment_below = None
            self.accompaniment_above = ["prefabs"]
            self.include_bass = True
        else:  # self.prefab_voice == "soprano"
            self.accompaniment_below = ["prefabs"]
            self.accompaniment_above = None
            self.include_bass = True
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


class PrefabComposer:
    def __init__(self, settings: t.Optional[PrefabComposerSettings] = None):
        if settings is None:
            settings = PrefabComposerSettings()
        self.structural_partitioner = StructuralPartitioner(settings)
        self.two_part_contrapuntist = TwoPartContrapuntist(settings)
        self.prefab_applier = PrefabApplier(settings)
        self.dumb_accompanist = DumbAccompanist(settings)
        self.settings = settings
        self._scales = ScaleDict()
        self._bass_range = settings.bass_range
        self._mel_range = settings.mel_range
        self.missing_prefabs = Counter()
        self._n_recurse_calls = 0
        self._spinner = it.cycle([char for char in r"/|\\-"])

    def _get_ranges(self, bass_range, mel_range):
        if bass_range is None:
            if self._bass_range is None:
                raise ValueError
            bass_range = self._bass_range
        if mel_range is None:
            if self._mel_range is None:
                raise ValueError
            mel_range = self._mel_range
        return bass_range, mel_range

    def _recurse(
        self,
        i: int,
        score: Score,
    ):
        if self._n_recurse_calls > self.settings.max_recurse_calls:
            logging.info(
                f"Max recursion calls {self.settings.max_recurse_calls} reached\n"
                + self.get_missing_prefab_str()
            )
            raise RecursionFailed(
                f"Max recursion calls {self.settings.max_recurse_calls} reached"
            )
        char = next(self._spinner)
        print(char, end="\r", flush=True)
        self._n_recurse_calls += 1
        if i:
            assert i - 1 == len(score.accompaniments) == len(score.prefabs)
        assert i == len(score.structural_melody)
        if i == len(score.chords):
            logging.debug(f"{self.__class__.__name__}._recurse: {i} final step")
            try:
                for prefab in self.prefab_applier._final_step(score):
                    with append_attempt(score.prefabs, prefab):
                        for pattern in self.dumb_accompanist._final_step(score):
                            with append_attempt(score.accompaniments, pattern):
                                return
            except MissingPrefabError as exc:
                self.missing_prefabs[str(exc)] += 1
            raise DeadEnd
        logging.debug(
            f"{self.__class__.__name__}._recurse: {i} {score.chords[i].token} "
            f"onset={score.chords[i].onset} "
            f"structural_bass={score.structural_bass[i]}"
        )
        # There should be two outcomes to the recursive stack:
        #   1. success
        #   2. a subclass of UndoRecursiveStep, in which case the append_attempt
        #       context manager handles popping from the list
        for mel_pitch in self.two_part_contrapuntist._step(score):
            try:
                with append_attempt(
                    score.structural_melody,
                    mel_pitch,
                    reraise=MissingPrefabError,
                ):
                    if i == 0:
                        # appending prefab requires at least two structural
                        #   melody pitches
                        return self._recurse(i + 1, score)
                    else:
                        for prefab in self.prefab_applier._step(score):
                            with append_attempt(score.prefabs, prefab):
                                for pattern in self.dumb_accompanist._step(
                                    score
                                ):
                                    with append_attempt(
                                        self._dumb_accompanist_target, pattern
                                    ):
                                        return self._recurse(i + 1, score)
            except MissingPrefabError as exc:
                self.missing_prefabs[str(exc)] += 1
        raise DeadEnd

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        bass_range: t.Optional[t.Tuple[int, int]] = None,
        mel_range: t.Optional[t.Tuple[int, int]] = None,
        return_ts: bool = False,
    ):
        """Args:
        chord_data: if string, should be in roman-text format.
            If a list, should be the output of the get_chords_from_rntxt
            function or similar."""
        self._n_recurse_calls = 0
        bass_range, mel_range = self._get_ranges(bass_range, mel_range)
        print("Reading score... ", end="", flush=True)
        score = Score(chord_data, bass_range, mel_range)
        print("done.")
        self.structural_partitioner(score)
        self.dumb_accompanist.init_new_piece(score.ts)
        if self.settings.accompaniment_annotations is AccompAnnots.NONE:
            self._dumb_accompanist_target = score.accompaniments
        else:
            dumb_accompanist_target = [score.accompaniments]
            if self.settings.accompaniment_annotations in (
                AccompAnnots.ALL,
                AccompAnnots.PATTERNS,
            ):
                dumb_accompanist_target.append(score.annotations["patterns"])
            if self.settings.accompaniment_annotations in (
                AccompAnnots.ALL,
                AccompAnnots.CHORDS,
            ):
                dumb_accompanist_target.append(score.annotations["chords"])
            # It's important that _dumb_accompanist_target be a tuple because
            #   this is how append_attempt() infers that it needs to unpack it
            #   (rather than append to it)
            self._dumb_accompanist_target = tuple(dumb_accompanist_target)
        try:
            self._recurse(
                0,
                score,
            )
        except DeadEnd:
            logging.info(
                f"Recursion reached a terminal dead end. Missing prefabs "
                "encountered along the way:\n" + self.get_missing_prefab_str()
            )
            raise RecursionFailed("Reached a terminal dead end")
        if self.settings.print_missing_prefabs:
            logging.info(
                "Completed score. Missing prefabs encountered along the way:\n"
                + self.get_missing_prefab_str()
            )
        if return_ts:
            return (
                score.get_df(["prefabs", "accompaniments", "annotations"]),
                score.ts.ts_str,
            )
        return score.get_df(["prefabs", "accompaniments", "annotations"])

    def get_missing_prefab_str(self, reverse=True, n=None):
        if reverse:
            outer_f = reversed
        else:
            outer_f = lambda x: x
        out = []
        for key, count in outer_f(self.missing_prefabs.most_common(n=n)):
            out.append(f"{count} failures:")
            out.append(key)
        return "\n".join(out)
