from collections import Counter
import typing as t
from dataclasses import dataclass

import pandas as pd

from .structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)

from .dumb_accompanist import DumbAccompanist, DumbAccompanistSettings

from .shared_classes import Note, Score

from .utils.recursion import DeadEnd, RecursionFailed, append_attempt

from dumb_composer.pitch_utils.rn_to_pc import rn_to_pc
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
    max_recurse_calls = 1000

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
            raise RecursionFailed(
                f"Max recursion calls {self.settings.max_recurse_calls} reached\n"
                + self.get_missing_prefab_str()
            )
        self._n_recurse_calls += 1
        if i == len(score.chords):
            for prefab in self.prefab_applier._final_step(score):
                with append_attempt(score.prefabs, prefab):
                    for pattern in self.dumb_accompanist._final_step(score):
                        with append_attempt(score.accompaniments, pattern):
                            return
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
                                        score.accompaniments, pattern
                                    ):
                                        return self._recurse(i + 1, score)
            except MissingPrefabError as exc:
                self.missing_prefabs[str(exc)] += 1

    def __call__(
        self,
        chord_data: t.Union[str, pd.DataFrame],
        bass_range: t.Optional[t.Tuple[int, int]] = None,
        mel_range: t.Optional[t.Tuple[int, int]] = None,
    ):
        """Args:
        chord_data: if string, should be in roman-text format.
            If a Pandas DataFrame, should be the output of the rn_to_pc
            function or similar."""
        self._n_recurse_calls = 0
        bass_range, mel_range = self._get_ranges(bass_range, mel_range)
        score = Score(chord_data, bass_range, mel_range)
        self.structural_partitioner(score)
        self.dumb_accompanist.init_new_piece(score.ts)
        try:
            self._recurse(
                0,
                score,
            )
        except DeadEnd:
            raise RecursionFailed(
                "Couldn't satisfy parameters\n" + self.get_missing_prefab_str()
            )
        return score.get_df(["prefabs", "accompaniments"])

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
