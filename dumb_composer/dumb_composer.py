import itertools as it
import logging
import random
import textwrap
import typing as t
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from dumb_composer.chord_spacer import NoSpacings, SimpleSpacer, SimpleSpacerSettings
from dumb_composer.constants import (
    DEFAULT_BASS_RANGE,
    DEFAULT_TENOR_ACCOMP_RANGE,
    DEFAULT_TENOR_MEL_RANGE,
)
from dumb_composer.dumb_accompanist import (
    AccompanimentDeadEnd,
    AccompAnnots,
    DumbAccompanist,
    DumbAccompanistSettings,
)
from dumb_composer.four_part_composer import FourPartComposerSettings, FourPartWorker
from dumb_composer.four_part_composer import (
    append_structural_pitches as append_four_part_pitches,
)
from dumb_composer.four_part_composer import (
    pop_structural_pitches as pop_four_part_pitches,
)
from dumb_composer.pitch_utils.chords import Chord
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.ranges import Ranger
from dumb_composer.pitch_utils.scale import ScaleDict
from dumb_composer.pitch_utils.spacings import RangeConstraints, SpacingConstraints
from dumb_composer.pitch_utils.types import ACCOMPANIMENTS, Pitch
from dumb_composer.pitch_utils.voice_lead_chords import voice_lead_chords
from dumb_composer.prefab_applier import (
    PrefabApplier,
    PrefabApplierSettings,
    PrefabDeadEnd,
    append_prefabs,
    pop_prefabs,
)
from dumb_composer.prefabs.prefab_pitches import MissingPrefabError
from dumb_composer.shared_classes import (
    AccompanimentInterface,
    FourPartScore,
    Note,
    PrefabInterface,
    PrefabScore,
)
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.time import Meter
from dumb_composer.two_part_contrapuntist import (
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.two_part_contrapuntist import (
    append_structural_pitches as append_two_part_pitches,
)
from dumb_composer.two_part_contrapuntist import (
    pop_structural_pitches as pop_two_part_pitches,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.recursion import (
    DeadEnd,
    RecursionFailed,
    StructuralDeadEnd,
    UndoRecursiveStep,
    append_attempt,
    recursive_attempt,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


@dataclass
class PrefabComposerSettings(
    DumbAccompanistSettings,
    PrefabApplierSettings,
    FourPartComposerSettings,
):
    bass_range: t.Optional[t.Tuple[int, int]] = None
    mel_range: t.Optional[t.Tuple[int, int]] = None
    max_recurse_calls: t.Optional[int] = None
    print_missing_prefabs: bool = True
    # if top_down_tie_prob is a single number, it sets a probability of tying
    #    melody notes through to the next chord anytime the pitch is repeated
    #   from one chord to the next. If it is a dictionary of form (int, float)
    #   it sets the probability of tying melody notes anytime the metric weight
    #   of the note that would begin the tie is greater than or equal to the
    #   next lesser key in the dict. (If there is no such key, then the note
    #   is not tied.)
    top_down_tie_prob: t.Optional[t.Union[float, t.Dict[int, float]]] = None
    structural_worker: t.Literal["two_part", "four_part"] = "four_part"

    def __post_init__(self):
        # We need to reconcile DumbAccompanistSettings'
        # settings with PrefabApplierSettings 'prefab_voice' setting.
        # TODO: (Malcolm 2023-07-29)
        # if (
        #     self.mel_range is None
        #     and self.accomp_range is None
        #     and self.bass_range is None
        # ):
        #     ranges = Ranger()(melody_part=self.prefab_voice)
        #     self.mel_range = ranges["mel_range"]
        #     self.bass_range = ranges["bass_range"]
        #     self.accomp_range = ranges["accomp_range"]
        LOGGER.debug(f"running PrefabComposerSettings __post_init__()")
        # TODO: (Malcolm 2023-07-29)
        # if self.prefab_voice == "bass":
        #     self.accompaniment_below = None
        #     self.accompaniment_above = ["prefabs"]
        #     self.include_bass = False
        #     if self.mel_range is None:
        #         self.mel_range = DEFAULT_BASS_RANGE
        # elif self.prefab_voice == "tenor":
        #     # TODO set ranges in a better/more dynamic way
        #     if self.mel_range is None:
        #         self.mel_range = DEFAULT_TENOR_MEL_RANGE
        #     if self.accomp_range is None:
        #         self.accomp_range = DEFAULT_TENOR_ACCOMP_RANGE
        #     self.accompaniment_below = None
        #     self.accompaniment_above = ["prefabs"]
        #     self.include_bass = True
        # else:  # self.prefab_voice == "soprano"
        #     self.accompaniment_below = ["prefabs"]
        #     self.accompaniment_above = None
        #     self.include_bass = True
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


class PrefabComposer:
    def __init__(self, settings: t.Optional[PrefabComposerSettings] = None):
        if settings is None:
            settings = PrefabComposerSettings()
        self.settings = settings
        self.structural_partitioner = StructuralPartitioner(settings)
        self._structural_worker: None | TwoPartContrapuntist | FourPartWorker = None

        if self.settings.structural_worker == "two_part":
            self._structural_worker_cls = TwoPartContrapuntist
            self._structural_append_func = append_two_part_pitches
            self._structural_pop_func = pop_two_part_pitches
        elif self.settings.structural_worker == "four_part":
            self._structural_worker_cls = FourPartWorker
            self._structural_append_func = append_four_part_pitches
            self._structural_pop_func = pop_four_part_pitches
        else:
            raise ValueError()

        self._prefab_applier: None | PrefabApplier = None
        self._dumb_accompanist: None | DumbAccompanist = None
        self.settings = settings
        LOGGER.debug(
            textwrap.fill(f"settings: {self.settings}", subsequent_indent=" " * 4)
        )
        self._scales = ScaleDict()
        self._bass_range = settings.bass_range
        self._mel_range = settings.mel_range
        self._n_recurse_calls = 0
        self._spinner = Spinner()

    def _get_ranges(
        self,
        bass_range: t.Optional[t.Tuple[int, int]],
        mel_range: t.Optional[t.Tuple[int, int]],
    ) -> t.Tuple[t.Tuple[int, int], t.Tuple[int, int]]:
        # TODO I think I can remove this function
        if bass_range is None:
            bass_range = self._bass_range
        if mel_range is None:
            mel_range = self._mel_range
        return bass_range, mel_range  # type:ignore

    def _check_n_recurse_calls(self):
        if (
            self.settings.max_recurse_calls is not None
            and self._n_recurse_calls > self.settings.max_recurse_calls
        ):
            LOGGER.info(
                f"Max recursion calls {self.settings.max_recurse_calls} reached\n"
                + self._prefab_applier.get_missing_prefab_str()  # type:ignore
            )
            raise RecursionFailed(
                f"Max recursion calls {self.settings.max_recurse_calls} reached"
            )

    def _final_step(self, i: int, score: PrefabScore):
        LOGGER.debug(f"{self.__class__.__name__}._recurse: {i} final step")
        assert self._prefab_applier is not None
        assert self._dumb_accompanist is not None

        for prefabs in self._prefab_applier._final_step():
            with recursive_attempt(
                do_func=append_prefabs,
                do_args=(prefabs, score),
                undo_func=pop_prefabs,
                undo_args=(prefabs, score),
            ):
                # with append_attempt(score.prefabs, prefab):
                # for pattern in self._dumb_accompanist._final_step():
                #     with append_attempt(score.accompaniments, pattern):
                return
        raise DeadEnd("reached end of _final_step()")

    def _validate_state(self, i: int, score: PrefabScore):
        # TODO: (Malcolm 2023-07-28) restore
        # == len(score.accompaniments) check
        try:
            assert i == 0
        except AssertionError:
            for prefabs in score.prefabs.values():
                assert len(prefabs) == i - 1
        assert i == len(score.structural_soprano)

    def _recurse(self, i: int, score: PrefabScore):
        self._check_n_recurse_calls()

        self._spinner()

        self._n_recurse_calls += 1

        self._validate_state(i, score)

        # final step
        if i == len(score.chords):
            self._final_step(i, score)
            return

        LOGGER.debug(
            f"{self.__class__.__name__}._recurse: {i=} {score.chords[i].token=} "
            f"{score.chords[i].onset=} "
            f"{score.pc_bass[i]=}"
        )

        assert self._structural_worker is not None
        assert self._prefab_applier is not None
        assert self._dumb_accompanist is not None
        # There should be two outcomes to the recursive stack:
        #   1. success
        #   2. a subclass of UndoRecursiveStep, in which case the append_attempt
        #       context manager handles popping from the list
        for pitches in self._structural_worker.step():
            LOGGER.debug(f"{pitches=}")
            with recursive_attempt(
                do_func=self._structural_append_func,
                do_args=(pitches, score),
                undo_func=self._structural_pop_func,
                undo_args=(score,),
            ):
                if i == 0:
                    # appending prefab requires at least two structural
                    #   melody pitches
                    return self._recurse(i + 1, score)
                else:
                    for prefabs in self._prefab_applier.step():
                        LOGGER.debug(f"{prefabs=}")
                        with recursive_attempt(
                            do_func=append_prefabs,
                            do_args=(prefabs, score),
                            undo_func=pop_prefabs,
                            undo_args=(prefabs, score),
                        ):
                            for pattern in self._dumb_accompanist.step(pitches):
                                LOGGER.debug(f"{pattern=}")
                                with append_attempt(
                                    self._dumb_accompanist_target,
                                    pattern,
                                ):
                                    return self._recurse(i + 1, score)

        raise DeadEnd("end of call to _recurse()")

    # TODO: (Malcolm) make this a function, write doctests
    def _apply_top_down_ties(self, score: PrefabScore):
        """
        Applies ties in a "top-down" manner.

        After the score is finished, we look at any pitches that are repeated
        from the end of one prefab to the beginning of the next, and tie them
        according to `settings.top_down_tie_prob`

        If top_down_tie_prob is a single number, it sets a probability of tying
        melody notes through to the next chord anytime the pitch is repeated
        from one chord to the next. If it is a dictionary of form (int, float)
        it sets the probability of tying melody notes anytime the metric weight
        of the note that would begin the tie is greater than or equal to the
        next lesser key in the dict. (If there is no such key, then the note
        is not tied.)

        I implemented this function some time after doing my main work on the
        rest of the script. It may be that there would be a way of achieving
        similar results that would be more integrated into the rest of the
        algorithm as opposed to this somewhat ad-hoc "top-down" approach.
        """
        top_down_tie_prob = self.settings.top_down_tie_prob
        is_float = isinstance(top_down_tie_prob, float)

        if not is_float:
            # We expect prob to be defined before use in the loop below but we define
            # it here to silence pylance warnings
            prob = 0.0

            # 10 should be higher than any metric weight we are likely to
            # observe in wild
            assert isinstance(top_down_tie_prob, dict)
            for i in range(min(top_down_tie_prob), 10):
                if i in top_down_tie_prob:
                    prob = top_down_tie_prob[i]
                else:
                    top_down_tie_prob[i] = prob
        for voice, notes in score.prefabs.items():
            for prefab1, prefab2 in zip(notes, notes[1:]):
                before_note = prefab1[-1]
                after_note = prefab2[0]
                if before_note.tie_to_next:
                    continue
                if before_note.pitch != after_note.pitch:
                    continue
                if before_note.release != after_note.onset:
                    continue
                if is_float:
                    if random.random() < top_down_tie_prob:
                        before_note.tie_to_next = True
                else:
                    note_weight = score.ts.weight(before_note.onset)
                    prob = top_down_tie_prob.get(note_weight, 0.0)
                    if random.random() < prob:
                        before_note.tie_to_next = True

    def __call__(
        self,
        chord_data: t.Union[str, t.List[Chord]],
        # bass_range: t.Optional[t.Tuple[int, int]] = None,
        # mel_range: t.Optional[t.Tuple[int, int]] = None,
        return_ts: bool = False,
        transpose: int = 0,
    ) -> pd.DataFrame:
        """Args:
        chord_data: if string, should be in roman-text format.
            If a list, should be the output of the get_chords_from_rntxt
            function or similar."""
        self._n_recurse_calls = 0

        print("Reading score... ", end="", flush=True)
        score = PrefabScore(chord_data, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self._structural_worker = self._structural_worker_cls(
            score=score, settings=self.settings
        )
        prefab_interface = PrefabInterface(score)
        self._prefab_applier = PrefabApplier(
            score_interface=prefab_interface, settings=self.settings
        )
        accompanist_interface = AccompanimentInterface(score)
        self._dumb_accompanist = DumbAccompanist(
            score_interface=accompanist_interface,
            settings=self.settings,
            voices_to_accompany=self._prefab_applier.decorated_voices,
        )
        if self.settings.accompaniment_annotations is AccompAnnots.NONE:
            self._dumb_accompanist_target = score.accompaniments
        else:
            raise NotImplementedError
            dumb_accompanist_target: t.List[t.Any] = [score.accompaniments]
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
            self._recurse(0, score)
        except DeadEnd:
            LOGGER.info(
                f"Recursion reached a terminal dead end. Missing prefabs "
                "encountered along the way:\n"
                + self._prefab_applier.get_missing_prefab_str()
            )
            raise RecursionFailed("Reached a terminal dead end")
        if self.settings.print_missing_prefabs:
            LOGGER.info(
                "Completed score. Missing prefabs encountered along the way:\n"
                + self._prefab_applier.get_missing_prefab_str()
            )
        if self.settings.top_down_tie_prob is not None:
            self._apply_top_down_ties(score)
        self._structural_worker = None
        if return_ts:
            # TODO: (Malcolm 2023-07-13) update type annotation?
            return (  # type:ignore
                score.get_df(["prefabs", "accompaniments", "annotations"]),
                score.ts.ts_str,
            )
        if self.settings.structural_worker == "two_part":
            structural_voices = ["structural_bass", "structural_soprano"]
        else:
            structural_voices = [
                "structural_bass",
                "structural_tenor",
                "structural_alto",
                "structural_soprano",
            ]
        return score.get_df(
            structural_voices
            + [
                "prefabs",
                "accompaniments",
                "annotations",  # TODO: (Malcolm 2023-08-01) restore
            ]
        )
