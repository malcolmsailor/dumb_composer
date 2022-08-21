from collections import defaultdict
from functools import cached_property
from numbers import Number
import pandas as pd
import typing as t

from .utils.df_helpers import sort_note_df

from .pitch_utils.scale import ScaleDict

from .pitch_utils.put_in_range import put_in_range

from .pitch_utils.rn_to_pc import rn_to_pc


class Annotation(pd.Series):
    def __init__(self, onset, text):
        super().__init__(
            {"onset": onset, "text": text, "type": "text", "track": 0}
        )


class Note(pd.Series):
    def __init__(
        self,
        pitch: int,
        onset: Number,
        release: t.Optional[Number] = None,
        dur: t.Optional[Number] = None,
        track: int = 1,
    ):
        if release is None:
            release = onset + dur
        super().__init__(
            {
                "pitch": pitch,
                "onset": onset,
                "release": release,
                "type": "note",
                "track": track,
            }
        )


def notes(
    pitches: t.Sequence[int],
    onset: Number,
    release: t.Optional[Number] = None,
    dur: t.Optional[Number] = None,
    track: int = 1,
) -> t.List[Note]:
    return [
        Note(pitch, onset, release=release, dur=dur, track=track)
        for pitch in pitches
    ]


# class Chord:
#     def __init__(
#         self,
#         pcs: t.Sequence[int],
#         scale_pcs: t.Sequence[int],
#         onset: Number,
#         release: Number,
#         foot: int,
#         token: t.Optional[str],
#     ):
#         # foot, bass, melody, pcs, inversion, scale, onset, release, inversion
#         pass


class ScaleGetter:
    def __init__(self, scale_pcs: t.Iterable[t.Sequence[int]]):
        self._scale_pcs = tuple(scale for scale in scale_pcs)
        self._scales = ScaleDict()

    def __getitem__(self, idx: int):
        return self._scales[self._scale_pcs[idx]]


class StructuralMelodyIntervalGetter:
    def __init__(
        self,
        scales: ScaleGetter,
        structural_melody: t.List[int],
        structural_bass: t.List[int],
    ):
        self.scales = scales
        self.structural_melody = structural_melody
        self.structural_bass = structural_bass
        self._melody_intervals: t.List[int] = []

    def __getitem__(self, idx):
        try:
            return self._melody_intervals[idx]
        except IndexError:
            for i in range(len(self._melody_intervals), idx + 1):
                self._melody_intervals.append(
                    self.scales[i].get_interval_class(
                        self.structural_bass[i], self.structural_melody[i]
                    )
                )
            return self._melody_intervals[idx]


class Score:
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.
    """

    def __init__(
        self,
        chord_data: t.Union[str, pd.DataFrame],
        bass_range: t.Tuple[int, int] = (30, 50),
        mel_range: t.Tuple[int, int] = (60, 78),
        prefab_track: int = 1,
        bass_track: int = 2,
        accompaniments_track: int = 3,
    ):
        if isinstance(chord_data, str):
            chord_data, _, ts = rn_to_pc(chord_data)
        self.ts = ts
        self.chord_data = chord_data
        self._scale_getter = ScaleGetter(chord_data.scale_pcs)
        self.bass_range = bass_range
        self.mel_range = mel_range
        self.structural_melody = []
        self._structural_melody_interval_getter = (
            StructuralMelodyIntervalGetter(
                self.scales, self.structural_melody, self.structural_bass
            )
        )
        self.contents = []
        self.prefabs = []
        self.accompaniments = []
        self.annotations = []
        self.prefab_track = prefab_track
        self.bass_track = bass_track
        self.accompaniments_track = accompaniments_track

    @cached_property
    def structural_bass(self) -> t.List[int]:
        return list(put_in_range(self.chord_data["foot"], *self.bass_range))

    @property
    def chords(self) -> pd.DataFrame:
        return self.chord_data

    @property
    def scales(self) -> ScaleGetter:
        return self._scale_getter

    @property
    def structural_melody_intervals(self) -> StructuralMelodyIntervalGetter:
        return self._structural_melody_interval_getter

    @property
    def structural_bass_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            Note(
                bass_pitch,
                self.chords.iloc[i].onset,
                self.chords.iloc[i].release,
                track=self.bass_track,
            )
            for i, bass_pitch in enumerate(self.structural_bass)
        )

    @property
    def prefabs_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(note for prefab in self.prefabs for note in prefab)

    @property
    def accompaniments_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            note
            for accompaniment in self.accompaniments
            for note in accompaniment
        )

    def get_df(self, contents: t.Union[str, t.Sequence[str]]) -> pd.DataFrame:
        if isinstance(contents, str):
            contents = [contents]
        dfs = (getattr(self, f"{name}_as_df") for name in contents)
        df = pd.concat(dfs)
        return sort_note_df(df)
