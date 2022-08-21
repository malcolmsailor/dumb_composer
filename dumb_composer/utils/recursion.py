import typing as t
from contextlib import contextmanager


@contextmanager
def append_attempt(list_: t.List[t.Any], item: t.Any):
    list_.append(item)
    try:
        yield
    except:
        list_.pop()
        raise
