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
    AccompAnnots,
    DumbAccompanist,
    DumbAccompanistSettings,
)
from dumb_composer.pitch_utils.chords import Chord
from dumb_composer.pitch_utils.intervals import IntervalQuerier
from dumb_composer.pitch_utils.ranges import Ranger
from dumb_composer.pitch_utils.scale import ScaleDict
from dumb_composer.pitch_utils.spacings import RangeConstraints, SpacingConstraints
from dumb_composer.pitch_utils.types import Pitch
from dumb_composer.pitch_utils.voice_lead_chords import voice_lead_chords
from dumb_composer.prefab_applier import PrefabApplier, PrefabApplierSettings
from dumb_composer.prefabs.prefab_pitches import MissingPrefabError
from dumb_composer.shared_classes import FourPartScore, Note, PrefabScore
from dumb_composer.structural_partitioner import (
    StructuralPartitioner,
    StructuralPartitionerSettings,
)
from dumb_composer.time import Meter
from dumb_composer.two_part_contrapuntist import (
    TwoPartContrapuntist,
    TwoPartContrapuntistSettings,
)
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.recursion import DeadEnd, RecursionFailed, append_attempt

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


@dataclass
class PrefabComposerSettings(
    TwoPartContrapuntistSettings,
    DumbAccompanistSettings,
    PrefabApplierSettings,
    StructuralPartitionerSettings,
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

    def __post_init__(self):
        # We need to reconcile DumbAccompanistSettings'
        # settings with PrefabApplierSettings 'prefab_voice' setting.
        if (
            self.mel_range is None
            and self.accomp_range is None
            and self.bass_range is None
        ):
            ranges = Ranger()(melody_part=self.prefab_voice)
            self.mel_range = ranges["mel_range"]
            self.bass_range = ranges["bass_range"]
            self.accomp_range = ranges["accomp_range"]
        logging.debug(f"running PrefabComposerSettings __post_init__()")
        if self.prefab_voice == "bass":
            self.accompaniment_below = None
            self.accompaniment_above = ["prefabs"]
            self.include_bass = False
            if self.mel_range is None:
                self.mel_range = DEFAULT_BASS_RANGE
        elif self.prefab_voice == "tenor":
            # TODO set ranges in a better/more dynamic way
            if self.mel_range is None:
                self.mel_range = DEFAULT_TENOR_MEL_RANGE
            if self.accomp_range is None:
                self.accomp_range = DEFAULT_TENOR_ACCOMP_RANGE
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
        self._two_part_contrapuntist: None | TwoPartContrapuntist = None
        self.prefab_applier = PrefabApplier(settings)
        self.dumb_accompanist = DumbAccompanist(settings)
        self.settings = settings
        logging.debug(
            textwrap.fill(f"settings: {self.settings}", subsequent_indent=" " * 4)
        )
        self._scales = ScaleDict()
        self._bass_range = settings.bass_range
        self._mel_range = settings.mel_range
        self.missing_prefabs = Counter()
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
            logging.info(
                f"Max recursion calls {self.settings.max_recurse_calls} reached\n"
                + self.get_missing_prefab_str()
            )
            raise RecursionFailed(
                f"Max recursion calls {self.settings.max_recurse_calls} reached"
            )

    def _final_step(self, i: int, score: PrefabScore):
        logging.debug(f"{self.__class__.__name__}._recurse: {i} final step")
        try:
            for prefab in self.prefab_applier._final_step(score):
                with append_attempt(score.prefabs, prefab):
                    for pattern in self.dumb_accompanist._final_step(score):
                        with append_attempt(score.accompaniments, pattern):
                            return
        except MissingPrefabError as exc:
            self.missing_prefabs[str(exc)] += 1
        raise DeadEnd()

    def _recurse(self, i: int, score: PrefabScore):
        self._check_n_recurse_calls()

        self._spinner()

        self._n_recurse_calls += 1

        assert (i == 0) or (i - 1 == len(score.accompaniments) == len(score.prefabs))
        assert i == len(score.structural_melody)

        # final step
        if i == len(score.chords):
            self._final_step(i, score)
            return

        logging.debug(
            f"{self.__class__.__name__}._recurse: {i=} {score.chords[i].token=} "
            f"{score.chords[i].onset=} "
            f"{score.pc_bass[i]=}"
        )

        assert self._two_part_contrapuntist is not None
        # There should be two outcomes to the recursive stack:
        #   1. success
        #   2. a subclass of UndoRecursiveStep, in which case the append_attempt
        #       context manager handles popping from the list
        for pitches in self._two_part_contrapuntist._step():
            try:
                with append_attempt(
                    (score.structural_bass, score.structural_melody),
                    (pitches["bass"], pitches["melody"]),
                    reraise=MissingPrefabError,
                ):
                    if i == 0:
                        # appending prefab requires at least two structural
                        #   melody pitches
                        return self._recurse(i + 1, score)
                    else:
                        for prefab in self.prefab_applier._step(score):
                            with append_attempt(score.prefabs, prefab):
                                for pattern in self.dumb_accompanist._step(score):
                                    with append_attempt(
                                        self._dumb_accompanist_target, pattern
                                    ):
                                        return self._recurse(i + 1, score)
            except MissingPrefabError as exc:
                self.missing_prefabs[str(exc)] += 1
        raise DeadEnd()

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
        for prefab1, prefab2 in zip(score.prefabs, score.prefabs[1:]):
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
        bass_range: t.Optional[t.Tuple[int, int]] = None,
        mel_range: t.Optional[t.Tuple[int, int]] = None,
        return_ts: bool = False,
        transpose: int = 0,
    ) -> pd.DataFrame:
        """Args:
        chord_data: if string, should be in roman-text format.
            If a list, should be the output of the get_chords_from_rntxt
            function or similar."""
        self._n_recurse_calls = 0
        range_constraints = RangeConstraints(
            min_bass_pitch=None if bass_range is None else bass_range[0],
            max_bass_pitch=None if bass_range is None else bass_range[1],
            min_melody_pitch=None if mel_range is None else mel_range[0],
            max_melody_pitch=None if mel_range is None else mel_range[1],
        )
        # TODO: (Malcolm 2023-07-20) we shouldn't be setting range_constraints here
        self.settings.range_constraints = range_constraints
        bass_range, mel_range = self._get_ranges(bass_range, mel_range)

        print("Reading score... ", end="", flush=True)
        score = PrefabScore(chord_data, range_constraints, transpose=transpose)
        print("done.")
        self.structural_partitioner(score)
        self.dumb_accompanist.init_new_piece(score.ts)
        self._two_part_contrapuntist = TwoPartContrapuntist(
            score=score, settings=self.settings
        )
        if self.settings.accompaniment_annotations is AccompAnnots.NONE:
            self._dumb_accompanist_target = score.accompaniments
        else:
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
        if self.settings.top_down_tie_prob is not None:
            self._apply_top_down_ties(score)
        self._two_part_contrapuntist = None
        if return_ts:
            # TODO: (Malcolm 2023-07-13) update type annotation?
            return (  # type:ignore
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
