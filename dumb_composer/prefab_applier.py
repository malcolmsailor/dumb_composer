from collections import defaultdict
from numbers import Number
import typing as t
import random
import warnings
import pandas as pd

from dumb_composer.pitch_utils.intervals import (
    get_relative_chord_factors,
)
from .pitch_utils.scale import Scale, ScaleDict
from dumb_composer.shared_classes import Note, Score

from dumb_composer.prefabs.prefab_pitches import (
    MissingPrefabError,
    PrefabPitchDirectory,
    PrefabPitches,
)
from dumb_composer.prefabs.prefab_rhythms import (
    PrefabRhythmDirectory,
    PrefabRhythms,
)


class PrefabApplier:
    prefab_rhythm_dir = PrefabRhythmDirectory()
    prefab_pitch_dir = PrefabPitchDirectory()

    def __init__(self, settings: t.Optional[t.Any] = None):
        # TODO settings
        pass

    def _append_from_prefabs(
        self,
        initial_pitch: int,
        offset: float,
        prefab_pitches: PrefabPitches,
        prefab_rhythms: PrefabRhythms,
        scale: Scale,
        track: int = 1,
    ):
        notes = []
        # need to displace by initial_onset somewhere
        orig_scale_degree = scale.index(initial_pitch)
        for rel_scale_degree, onset, release in zip(
            prefab_pitches.relative_degrees,
            prefab_rhythms.onsets,
            prefab_rhythms.releases,
        ):
            new_pitch = scale[orig_scale_degree + rel_scale_degree]
            notes.append(
                Note(new_pitch, onset + offset, release + offset, track=track)
            )
        return notes

    def _step(
        self,
        score: Score,
    ):
        current_i = len(score.prefabs)
        current_scale = score.scales[current_i]
        next_scale = score.scales[current_i + 1]
        current_chord = score.chords.iloc[current_i]
        current_mel_pitch = score.structural_melody[current_i]
        next_mel_pitch = score.structural_melody[current_i + 1]
        rhythm_options = self.prefab_rhythm_dir(
            current_chord.release
            - current_chord.onset
            # TODO metric strength
        )
        while rhythm_options:
            rhythm_i = random.randrange(len(rhythm_options))
            rhythm = rhythm_options.pop(rhythm_i)
            generic_pitch_interval = current_scale.get_interval(
                current_mel_pitch, next_mel_pitch, scale2=next_scale
            )
            relative_chord_factors = get_relative_chord_factors(
                score.structural_melody_intervals[current_i],
                current_chord.intervals_above_bass,
                len(current_scale),
            )
            pitch_options = self.prefab_pitch_dir(
                generic_pitch_interval,
                rhythm.metric_strength_str,
                relative_chord_factors,
            )
            while pitch_options:
                pitch_i = random.randrange(len(pitch_options))
                pitches = pitch_options.pop(pitch_i)
                yield self._append_from_prefabs(
                    current_mel_pitch,
                    current_chord.onset,
                    pitches,
                    rhythm,
                    current_scale,
                    track=score.prefab_track,
                )

    def __call__(
        self,
        score: Score,
    ):
        warnings.warn(
            "deprecated in favor of PrefabComposer", DeprecationWarning
        )
        # missing_prefab_errors = []
        # for _ in range(len(score.structural_melody)):
        #     try:
        #         score.prefabs.extend(next(self._step(score)))
        #     except MissingPrefabError as exc:
        #         missing_prefab_errors.append(exc)
        #         continue
        # if missing_prefab_errors:
        #     for e in missing_prefab_errors:
        #         print(e)
        #     raise ValueError
        # return pd.DataFrame(score.prefabs)
