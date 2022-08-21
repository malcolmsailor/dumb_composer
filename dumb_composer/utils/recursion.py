import typing as t
from contextlib import contextmanager


class RecursionFailed(Exception):
    pass


class UndoRecursiveStep(Exception):
    pass


class DeadEnd(UndoRecursiveStep):
    pass


@contextmanager
def append_attempt(
    list_: t.List[t.Any],
    item: t.Any,
    reraise: t.Optional[
        t.Union[t.Type[UndoRecursiveStep], t.Tuple[t.Type[UndoRecursiveStep]]]
    ] = None,
):
    list_.append(item)
    try:
        yield
    except UndoRecursiveStep as exc:
        list_.pop()
        if reraise is not None and isinstance(exc, reraise):
            raise
