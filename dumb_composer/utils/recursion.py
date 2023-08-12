import logging
import typing as t
from contextlib import contextmanager
from copy import deepcopy
from types import MappingProxyType

LOGGER = logging.getLogger(__name__)


class RecursionFailed(Exception):
    pass


class UndoRecursiveStep(Exception):
    pass


class DeadEnd(UndoRecursiveStep):
    def __init__(
        self,
        msg: str = "",
        save_deadends_to: list[t.Any] | None = None,
        max_deadends_to_save: int = 100,
        **kwargs,
    ):
        super().__init__(msg)
        if save_deadends_to is not None:
            if len(save_deadends_to) < max_deadends_to_save:
                save_deadends_to.append(deepcopy(kwargs))


class StructuralDeadEnd(DeadEnd):
    pass


@contextmanager
def append_attempt(
    list_: t.Union[t.List[t.Any], t.Tuple[t.List[t.Any,], ...]],
    item: t.Union[t.Any, t.Tuple[t.Any, ...]],
    # reraise: t.Optional[
    #     t.Union[t.Type[UndoRecursiveStep], t.Tuple[t.Type[UndoRecursiveStep], ...]]
    # ] = None,
    reraise_if_not: tuple[t.Type[UndoRecursiveStep]] | None = None,
):
    if isinstance(list_, tuple):
        for sub_list, sub_item in zip(list_, item):
            # LOGGER.debug(f"appending to sub_list {id(sub_list)} {id_}")
            sub_list.append(sub_item)
    else:
        # LOGGER.debug(f"appending to list {id(list)} {id_}")
        list_.append(item)
    try:
        yield
    except UndoRecursiveStep as exc:
        LOGGER.debug(f"undoing append attempt")
        LOGGER.debug(f"{exc.__class__.__name__}: {str(exc)}")
        if isinstance(list_, tuple):
            for sub_list in list_:
                # LOGGER.debug(f"popping from sub_list {id(sub_list)} {id_}")
                sub_list.pop()
        else:
            # LOGGER.debug(f"popping from list {id(list)} {id_}")
            list_.pop()
        if reraise_if_not is not None and not isinstance(exc, reraise_if_not):
            raise
        # if reraise is not None and isinstance(exc, reraise):
        #     raise


@contextmanager
def recursive_attempt(
    *,
    do_func: t.Callable,
    undo_func: t.Callable,
    do_args: t.Sequence[t.Any] = (),
    do_kwargs: t.Mapping[str, t.Any] = MappingProxyType({}),
    undo_args: t.Sequence[t.Any] = (),
    undo_kwargs: t.Mapping[str, t.Any] = MappingProxyType({}),
    # reraise: None
    # | t.Type[UndoRecursiveStep]
    # | tuple[t.Type[UndoRecursiveStep], ...] = None,
    reraise_if_not: tuple[t.Type[UndoRecursiveStep]] | None = None,
):
    LOGGER.debug(f"making recursive attempt {do_func=} {do_args=}")
    do_func(*do_args, **do_kwargs)
    try:
        yield
    except UndoRecursiveStep as exc:
        LOGGER.debug(f"undoing recursive attempt {undo_func=} {undo_args=}")
        LOGGER.debug(f"{exc.__class__.__name__}: {str(exc)}")
        undo_func(*undo_args, **undo_kwargs)
        # if reraise is not None and isinstance(exc, reraise):
        #     raise
        if reraise_if_not and not isinstance(exc, reraise_if_not):
            raise
