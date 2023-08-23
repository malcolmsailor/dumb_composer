import logging
import typing as t
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from functools import cached_property

import pandas as pd

from dumb_composer.constants import TRACKS
from dumb_composer.pitch_utils.chords import Allow, Chord, get_chords_from_rntxt
from dumb_composer.pitch_utils.music21_handler import get_ts_from_rntxt
from dumb_composer.pitch_utils.put_in_range import put_in_range  # used in doctests
from dumb_composer.pitch_utils.types import (
    TIME_TYPE,
    Annotation,
    InnerVoice,
    Note,
    OuterVoice,
    Pitch,
    TimeStamp,
    Voice,
)
from dumb_composer.shared_classes import (
    ScaleGetter,
    StructuralMelodyIntervals,
    apply_ties,
)
from dumb_composer.suspensions import Suspension
from dumb_composer.time import Meter
from dumb_composer.utils.df_helpers import sort_note_df
from dumb_composer.utils.iterables import flatten_iterables

LOGGER = logging.getLogger(__name__)


class _ScoreBase:
    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: Meter | str | None = None,
        transpose: int = 0,
    ):
        if isinstance(chord_data, str):
            logging.debug(f"reading chords from {chord_data}")
            ts = get_ts_from_rntxt(chord_data)
            chord_data = get_chords_from_rntxt(chord_data)
        elif ts is None:
            raise ValueError(f"`ts` must be supplied if `chord_data` is not a string")
        if isinstance(ts, str):
            self.ts = Meter(ts)
        else:
            self.ts = ts
        self._chords = chord_data
        if transpose:
            self._chords = [chord.transpose(transpose) for chord in self._chords]
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in chord_data)
        # self.range_constraints = range_constraints
        self._structural: defaultdict[Voice, list[Pitch]] = defaultdict(list)
        self._suspensions: defaultdict[
            Voice, dict[TimeStamp, Suspension]
        ] = defaultdict(dict)
        # TODO: (Malcolm 2023-08-04) I'm not sure if using the "structural unit" as
        #   key for suspension resolutions (quasi-bar number) still works. What if
        #   a soprano suspension leads to splitting a chord and then a tenor
        #   suspension leads to splitting the chord still further?
        self._suspension_resolutions: t.DefaultDict[
            Voice, t.Dict[TimeStamp, int]
        ] = defaultdict(dict)
        # self.annotations: defaultdict[str, t.List[Annotation]] = defaultdict(list)
        self.annotations: defaultdict[
            Voice, defaultdict[str, dict[TimeStamp, Annotation]]
        ] = defaultdict(lambda: defaultdict(dict))
        self.misc: dict[str, t.Any] = {}

        self._structural_soprano_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_soprano, self.structural_bass
        )
        # _split_chords indices are to the chord *before* the split. E.g., if we
        #   split chord i=3, the split chords will have i=3 and i=4. We then add 3
        #   to _split_chords.
        self._split_chords: Counter[int] = Counter()

    def __eq__(self, other):
        if type(other) is type(self):
            for k, v in self.__dict__.items():
                if k in ("_scale_getter"):
                    continue
                other_v = other.__dict__[k]
                if isinstance(v, defaultdict):
                    if v == other_v or (
                        not any(x for x in v.values())
                        and not any(x for x in other_v.values())
                    ):
                        continue
                    return False
                elif v != other.__dict__[k]:
                    return False
            return True
        return False

    # ----------------------------------------------------------------------------------
    # State validation etc.
    # ----------------------------------------------------------------------------------

    def validate_state(self) -> bool:
        return len({len(pitches) for pitches in self._structural.values()}) == 1

    # ----------------------------------------------------------------------------------
    # Properties and helper functions
    # ----------------------------------------------------------------------------------
    @cached_property
    def structural_soprano(self):
        return self._structural[OuterVoice.MELODY]

    @cached_property
    def structural_bass(self):
        return self._structural[OuterVoice.BASS]

    @cached_property
    def pc_bass(self) -> t.List[int]:
        return [chord.foot for chord in self._chords]  # type:ignore

    # TODO: (Malcolm 2023-08-10) possibly restore
    def get_suspension_at(self, time: TimeStamp, voice: Voice) -> Suspension | None:
        return self._suspensions[voice].get(time, None)

    def is_suspension_at(self, time: TimeStamp, voice: Voice) -> bool:
        return self.get_suspension_at(time, voice) is not None

    def is_resolution_at(self, time: TimeStamp, voice: Voice) -> bool:
        return time in self._suspension_resolutions[voice]

    @property
    def chords(self) -> t.List[Chord]:
        return self._chords

    @chords.setter
    def chords(self, new_chords: t.List[Chord]):
        self._chords = new_chords
        # deleting cached property self.pc_bass allows it to be regenerated next time it
        #   is needed
        if hasattr(self, "pc_bass"):
            del self.pc_bass
        # TODO: (Malcolm 2023-07-28) why is ScaleGetter needed? perhaps Scale
        #   should be composed within Chord?
        self._scale_getter = ScaleGetter(chord.scale_pcs for chord in self._chords)
        self._structural_soprano_interval_getter = StructuralMelodyIntervals(
            self.scales, self.structural_soprano, self.structural_bass
        )

    @property
    def scales(self) -> ScaleGetter:
        return self._scale_getter

    @property
    @abstractmethod
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError

    def get_existing_pitches(
        self,
        idx: int,
        attr_names: t.Sequence[str] | None = None,
    ) -> t.Tuple[Pitch]:
        raise NotImplementedError
        if attr_names is None:
            attr_names = self.default_existing_pitch_attr_names

        return tuple(  # type:ignore
            flatten_iterables(
                getattr(self, attr_name)[idx]
                for attr_name in attr_names
                if len(getattr(self, attr_name)) > idx
            )
        )

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TimeStamp):
        assert chord.onset < time < chord.release

    def split_ith_chord_at(
        self, i: int, time: TimeStamp, check_correctness: bool = True
    ) -> None:
        """
        >>> rntxt = '''Time Signature: 4/4
        ... m1 C: I
        ... m2 V
        ... m3 I'''
        >>> score = PrefabScore(rntxt)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        >>> score.split_ith_chord_at(1, 6.0)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 6.0: 'V', 8.0: 'I'}
        >>> score.merge_ith_chords(1)
        >>> {float(chord.onset): chord.token for chord in score.chords}
        {0.0: 'C:I', 4.0: 'V', 8.0: 'I'}
        """
        # just in case pc_bass has not been computed yet, we need
        #   to compute it now:
        LOGGER.debug(f"splitting chord {i=} at {time=}")
        self.pc_bass
        # TODO make debug flag for check_correctness
        chord = self.chords[i]
        if check_correctness:
            self._validate_split_ith_chord_at(i, chord, time)
        new_chord = chord.copy()
        chord.release = time
        new_chord.onset = time
        self.chords.insert(i + 1, new_chord)
        self.pc_bass.insert(i + 1, self.pc_bass[i])

        # Extend structural notes if necessary by duplicating the ith pitch
        for structural_notes in self._structural.values():
            if len(structural_notes) > i:
                structural_notes.insert(i + 1, structural_notes[i])

        # If we split a suspension, we want to copy it over to the new chord
        for suspensions in self._suspensions.values():
            if chord.onset in suspensions:
                suspensions[new_chord.onset] = suspensions[chord.onset]

        self._scale_getter.insert_scale_pcs(i + 1, new_chord.scale_pcs)
        self._split_chords[i] += 1

    def merge_ith_chords(self, i: int, check_correctness: bool = True) -> None:
        """
        Merge score.chords[i] and score.chords[i + 1]
        """
        # just in case pc_bass has not been computed yet, we need
        #   to compute it now:
        LOGGER.debug(f"merging chord {i=}")
        self.pc_bass

        chord1, chord2 = self.chords[i : i + 2]
        chord1.release = chord2.release
        self.chords.pop(i + 1)
        self.pc_bass.pop(i + 1)

        for structural_notes in self._structural.values():
            if len(structural_notes) > i + 1:
                structural_notes.pop(i + 1)

        # If there was a suspension at chord2 onset (e.g., because a suspension on
        #   chord1 was previously split), we need to remove it
        for suspensions in self._suspensions.values():
            if chord2.onset in suspensions:
                suspensions.pop(chord2.onset)

        # if len(self.structural_soprano) > i + 2:
        #     self.structural_soprano.pop(i + 1)

        self._scale_getter.pop_scale_pcs(i + 1)
        if check_correctness:
            assert self._split_chords[i]
            chord2.onset = chord1.onset
            assert chord1 == chord2

        self._split_chords[i] -= 1

    def get_flat_list_of_all_annotations(self) -> list[Annotation]:
        out = []
        for voice_annots in self.annotations.values():
            for annots_of_type in voice_annots.values():
                out.extend(list(annots_of_type.values()))

        return out

    @property
    def annotations_as_df(self) -> pd.DataFrame:
        out = pd.DataFrame(self.get_flat_list_of_all_annotations())
        # only one annotation per time-point appears in the kern files (or is it
        #   the verovio realizations?). Anyway, we merge them into one here. TODO
        #   is there a way around this constraint?
        if not len(out):
            return out
        temp = []
        out = out.sort_values(by=["onset", "track"])
        for (onset, track), rows in out.groupby(["onset", "track"]):
            new_annot = Annotation(onset, text="".join(rows["text"]), track=track)
            temp.append(new_annot)
        # for onset in sorted(out.onset.unique()):
        #     new_annot = Annotation(
        #         onset,
        #         "".join(annot.text for _, annot in out[out.onset == onset].iterrows()),
        #     )
        #     temp.append(new_annot)
        out = pd.DataFrame(temp)
        return out

    def as_df(self, attr_name: str, track: int = 0) -> pd.DataFrame:
        notes = getattr(self, attr_name)
        return pd.DataFrame(
            Note(
                melody_pitch,
                self.chords[i].onset,
                self.chords[i].release,
                track=track,
            )
            for i, melody_pitch in enumerate(notes)
        )

    @property
    def structural_melody_as_df(self):
        return self.as_df(
            "structural_melody", track=TRACKS["structural"][OuterVoice.MELODY]
        )

    @property
    def structural_bass_as_df(self):
        return self.as_df(
            "structural_bass", track=TRACKS["structural"][OuterVoice.BASS]
        )

    def get_df(self, contents: t.Union[str, t.Sequence[str]]) -> pd.DataFrame:
        if isinstance(contents, str):
            contents = [contents]
        dfs = []
        for name in contents:
            if hasattr(self, f"{name}_as_df"):
                dfs.append(getattr(self, f"{name}_as_df"))
            else:
                dfs.append(self.as_df(name))
        df = pd.concat(dfs)
        return sort_note_df(df)

    @property
    def structural_as_df(self) -> pd.DataFrame:
        dfs = []
        for voice, notes in self._structural.items():
            track = TRACKS["structural"][voice]
            sub_df = pd.DataFrame(
                Note(
                    pitch,
                    self.chords[i].onset,
                    self.chords[i].release,
                    track=track,
                )
                for i, pitch in enumerate(notes)
            )
            dfs.append(sub_df)
        df = pd.concat(dfs)
        return sort_note_df(df)


class Score(_ScoreBase):
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.

    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = Score(chord_data=rntxt)

    >>> [chord.pcs for chord in score.chords]
    [(0, 4, 7), (5, 9, 2), (7, 11, 2), (4, 7, 0)]

    >>> score.pc_bass  # Pitch classes
    [0, 5, 7, 4]
    """

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError
        return "structural_bass", "structural_soprano"  # type:ignore


class FourPartScore(_ScoreBase):
    """
    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = FourPartScore(chord_data=rntxt)

    >>> score.pc_bass  # Pitch-classes, not pcs
    [0, 5, 7, 4]

    Let's create a bass:
    >>> score.structural_bass.extend(put_in_range(pc, low=36) for pc in (score.pc_bass))
    >>> score.structural_bass
    [36, 41, 43, 40]

    Let's add a melody:
    >>> score.structural_soprano.extend([72, 74, 71, 72])

    # TODO: (Malcolm 2023-08-03) implement or change
    # >>> score.get_existing_pitches(0)
    # (36, 72)
    # >>> score.get_existing_pitches(1)
    # (41, 74)

    Let's add inner parts:
    >>> score.structural_tenor.extend([55, 57, 55, 55])
    >>> score.structural_alto.extend([64, 65, 62, 77])

    # >>> score.get_existing_pitches(0)
    # (36, 55, 64, 72)
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(
            chord_data,
            ts=ts,
            transpose=transpose,
        )

    @property
    def structural_tenor(self):
        return self._structural[InnerVoice.TENOR]

    @property
    def structural_alto(self):
        return self._structural[InnerVoice.ALTO]

    @property
    def structural_tenor_as_df(self):
        return self.as_df(
            "structural_tenor", track=TRACKS["structural"][InnerVoice.TENOR]
        )

    @property
    def structural_alto_as_df(self):
        return self.as_df(
            "structural_alto", track=TRACKS["structural"][InnerVoice.ALTO]
        )

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str, ...]:
        raise NotImplementedError
        return (
            "structural_bass",
            "structural_tenor",
            "structural_alto",
            "structural_soprano",
        )


class ScoreWithAccompaniments(ABC, _ScoreBase):
    @property
    @abstractmethod
    def accompaniments(self) -> list[list[Note]]:
        raise NotImplementedError()


class PrefabScore(FourPartScore):
    """This class provides a "shared working area" for the various classes and
    functions that build a score. It doesn't encapsulate much of anything.

    >>> rntxt = "m1 C: I b2 ii6 b3 V b4 I6"
    >>> score = PrefabScore(chord_data=rntxt)

    >>> [chord.pcs for chord in score.chords]
    [(0, 4, 7), (5, 9, 2), (7, 11, 2), (4, 7, 0)]

    >>> score.pc_bass  # Pitch-classes
    [0, 5, 7, 4]
    """

    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(chord_data, ts=ts, transpose=transpose)

        self.prefabs: defaultdict[Voice, list[list[Note]]] = defaultdict(list)
        self._tied_prefab_indices: defaultdict[Voice, set[int]] = defaultdict(set)
        self._allow_prefab_start_with_rest: t.Dict[int, Allow] = {}

    @property
    def default_existing_pitch_attr_names(self) -> t.Tuple[str]:
        raise NotImplementedError
        return "structural_bass", "structural_soprano"  # type:ignore

    @property
    def prefabs_as_df(self) -> pd.DataFrame:
        sub_dfs = []
        for voice, notes in self.prefabs.items():
            track = TRACKS["prefabs"][voice]
            tied_notes = apply_ties(
                (note for prefab in notes for note in prefab),
                check_correctness=True,  # TODO remove this when I'm more confident
            )
            sub_df = pd.DataFrame(tied_notes)
            sub_df["track"] = track
            sub_dfs.append(sub_df)
        return pd.concat(sub_dfs)

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TIME_TYPE):
        super()._validate_split_ith_chord_at(i, chord, time)
        for voice in self.prefabs:
            assert len(self.prefabs[voice]) <= i


class PrefabScoreWithAccompaniments(PrefabScore, ScoreWithAccompaniments):
    def __init__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        ts: t.Optional[t.Union[Meter, str]] = None,
        transpose: int = 0,
    ):
        super().__init__(chord_data, ts=ts, transpose=transpose)
        self._accompaniments: t.List[t.List[Note]] = []

    @property
    def accompaniments(self) -> list[list[Note]]:
        return self._accompaniments

    @property
    def accompaniments_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            note
            for accompaniment in self.accompaniments
            for note in accompaniment  # type:ignore
        )

    def _validate_split_ith_chord_at(self, i: int, chord: Chord, time: TIME_TYPE):
        super()._validate_split_ith_chord_at(i, chord, time)
        assert len(self.accompaniments) <= i
