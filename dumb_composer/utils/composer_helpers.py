import logging
from typing import Callable, Iterable, Sequence

from dumb_composer.pitch_utils.types import RecursiveWorker
from dumb_composer.utils.display import Spinner
from dumb_composer.utils.recursion import DeadEnd

LOGGER = logging.getLogger(__name__)


def chain_steps(recursive_workers: Iterable[RecursiveWorker]):
    all_workers = list(recursive_workers)
    spinner = Spinner()

    def _recurse_through_workers(
        workers: list[RecursiveWorker],
    ):
        spinner()
        if not workers:
            if all(w.finished for w in all_workers):
                yield
                return
            workers = all_workers

        worker, *remaining_workers_at_this_step = workers

        if remaining_workers_at_this_step:
            assert worker.step_i >= max(
                w.step_i for w in remaining_workers_at_this_step
            )

        # It is possible for some workers to get ahead of others. For example, if a
        #   later contrapuntist creates a suspension, it may split the notes in
        #   the prior voices of earlier contrapuntists. In this case, the next
        #   notes of the earlier contrapuntists already exist, and so we need to skip
        #   them.
        if (
            (not worker.ready)
            or worker.finished
            # or (
            #     remaining_workers_at_this_step
            #     and worker.step_i
            #     > min(w.step_i for w in remaining_workers_at_this_step)
            # )
        ):
            yield from _recurse_through_workers(remaining_workers_at_this_step)
            raise DeadEnd()

        for result in worker.step():
            LOGGER.debug(f"{worker.__class__.__name__} making append attempt")
            with worker.append_attempt(result):
                yield from _recurse_through_workers(remaining_workers_at_this_step)
        raise DeadEnd()

    # _recurse_through_workers is a generator so we need to call next() on it
    next(_recurse_through_workers(all_workers))
    spinner(terminate=False)
