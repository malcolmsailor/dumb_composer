from collections import Counter
import typing as t
from dataclasses import dataclass

import pandas as pd

from .dumb_accompanist import DumbAccompanist, DumbAccompanistSettings

from .shared_classes import Note, Score

from .utils.recursion import append_attempt

from dumb_composer.pitch_utils.rn_to_pc import rn_to_pc
from dumb_composer.pitch_utils.scale import ScaleDict
from dumb_composer.prefabs.prefab_pitches import MissingPrefabError
from dumb_composer.two_part_contrapuntist import (
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.prefab_applier import PrefabApplier


@dataclass
class PrefabComposerSettings(
    TwoPartContrapuntistSettings, DumbAccompanistSettings
):
    accompaniment_below: t.Optional[t.Union[str, t.Sequence[str]]] = "prefabs"


class PrefabComposer:
    def __init__(self, settings: t.Optional[PrefabComposerSettings] = None):
        if settings is None:
            settings = PrefabComposerSettings()
        self.two_part_contrapuntist = TwoPartContrapuntist(settings)
        self.prefab_applier = PrefabApplier(settings)  # TODO
        self.dumb_accompanist = DumbAccompanist(settings)
        self._scales = ScaleDict()
        self._bass_range = settings.bass_range
        self._mel_range = settings.mel_range
        self.missing_prefabs = Counter()

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
        if i == len(score.chords):
            # TODO need to append final melody notes
            return
        for mel_pitch in self.two_part_contrapuntist._step(score):
            try:
                with append_attempt(score.structural_melody, mel_pitch):
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
        bass_range, mel_range = self._get_ranges(bass_range, mel_range)
        score = Score(chord_data, bass_range, mel_range)
        self.dumb_accompanist.init_new_piece(score.ts)
        self._recurse(
            0,
            score,
        )
        return score.get_df(["structural_bass", "prefabs", "accompaniments"])

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
