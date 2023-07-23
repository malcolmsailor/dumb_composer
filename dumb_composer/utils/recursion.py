import logging
import typing as t
from contextlib import contextmanager
from copy import deepcopy


class RecursionFailed(Exception):
    pass


class UndoRecursiveStep(Exception):
    pass


class DeadEnd(UndoRecursiveStep):
    def __init__(
        self,
        save_deadends_to: list[t.Any] | None = None,
        max_deadends_to_save: int = 100,
        **kwargs,
    ):
        if save_deadends_to is not None:
            if len(save_deadends_to) < max_deadends_to_save:
                save_deadends_to.append(deepcopy(kwargs))


@contextmanager
def append_attempt(
    list_: t.Union[t.List[t.Any], t.Tuple[t.List[t.Any,], ...]],
    item: t.Union[t.Any, t.Tuple[t.Any, ...]],
    reraise: t.Optional[
        t.Union[t.Type[UndoRecursiveStep], t.Tuple[t.Type[UndoRecursiveStep], ...]]
    ] = None,
):
    if isinstance(list_, tuple):
        for sub_list, sub_item in zip(list_, item):
            # logging.debug(f"appending to sub_list {id(sub_list)} {id_}")
            sub_list.append(sub_item)
    else:
        # logging.debug(f"appending to list {id(list)} {id_}")
        list_.append(item)
    try:
        yield
    except UndoRecursiveStep as exc:
        logging.debug(f"{exc.__class__.__name__}: {str(exc)}")
        if isinstance(list_, tuple):
            for sub_list in list_:
                # logging.debug(f"popping from sub_list {id(sub_list)} {id_}")
                sub_list.pop()
        else:
            # logging.debug(f"popping from list {id(list)} {id_}")
            list_.pop()
        if reraise is not None and isinstance(exc, reraise):
            raise
