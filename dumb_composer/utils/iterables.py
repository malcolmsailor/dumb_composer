import itertools
import random
from typing import Any, Iterable, Iterator, Sequence, TypeVar

from dumb_composer.utils.math_ import softmax

T = TypeVar("T")


def unique_items_in_order(items: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(items))


def _flatten_iterables_sub(x: Any | Iterable[Any]) -> list[Any]:
    if isinstance(x, Iterable):
        out = []
        for y in x:
            out += _flatten_iterables_sub(y)
        return out
    else:
        return [x]


def flatten_iterables(l: Iterable[Any]) -> list[Any]:
    """
    >>> flatten_iterables([1, 2, 3])
    [1, 2, 3]
    >>> flatten_iterables(())
    []
    >>> flatten_iterables([1, (2, 3), [[4], [], [(i for i in range(5, 7))]]])
    [1, 2, 3, 4, 5, 6]
    """
    return _flatten_iterables_sub(l)


def shuffled_cartesian_product(*args, repeat=1) -> Iterable[Any]:
    """
    # TODO: (Malcolm 2023-07-12) it would be nice/fun to make a lazy version that
    doesn't store the entire product in memory in order to shuffle it.

    >>> list(shuffled_cartesian_product("ABCD", "xy"))  # doctest: +SKIP
    [('C', 'y'), ('B', 'x'), ('B', 'y'), ('A', 'y'), ('A', 'x'), ('C', 'x'),
    ('D', 'y'), ('D', 'x')]

    >>> list(shuffled_cartesian_product("A"))
    [('A',)]
    >>> list(shuffled_cartesian_product())
    [()]
    """
    out = list(itertools.product(*args, repeat=repeat))
    random.shuffle(out)
    yield from out


def yield_from_sequence_of_iters(
    iter_seq: Sequence[Iterable[T]], shuffle: bool = False
) -> Iterator[T]:
    """
    >>> iter1 = range(3)
    >>> iter2 = range(3, 7)
    >>> list(yield_from_sequence_of_iters([iter1, iter2]))
    [0, 3, 1, 4, 2, 5, 6]

    >>> iter3 = []
    >>> list(yield_from_sequence_of_iters([iter2, iter1, iter3]))
    [3, 0, 4, 1, 5, 2, 6]

    >>> list(
    ...     yield_from_sequence_of_iters([iter1, iter2], shuffle=True)
    ... )  # doctest: +SKIP
    [3, 4, 0, 1, 2, 5, 6]
    """
    iter_list = list(iter(i) for i in iter_seq)

    if shuffle:
        while iter_list:
            i = random.randrange(len(iter_list))
            try:
                yield next(iter_list[i])
            except StopIteration:
                iter_list.pop(i)
    else:
        while iter_list:
            to_pop = []
            for i, iter_ in enumerate(iter_list):
                try:
                    yield next(iter_)
                except StopIteration:
                    to_pop.append(i)
            for i in reversed(to_pop):
                iter_list.pop(i)


def yield_sample_from_sequence_of_iters(
    iter_seq: Sequence[Iterable[T]],
    weights: Sequence[float],
    apply_softmax: bool = True,
) -> Iterator[T]:
    """
    >>> seq1 = [1, 2, 3]
    >>> seq2 = "abcd"
    >>> weights = [5.0, 1.0]
    >>> list(
    ...     yield_sample_from_sequence_of_iters([seq1, seq2], weights=weights)
    ... )  # doctest: +SKIP
    [1, 2, 3, 'a', 'b', 'c', 'd']
    >>> weights = [1.0, 1.0]
    >>> list(
    ...     yield_sample_from_sequence_of_iters([seq1, seq2], weights=weights)
    ... )  # doctest: +SKIP
    ['a', 1, 2, 3, 'b', 'c', 'd']
    """
    if apply_softmax:
        weights = softmax(weights)

    weights = list(weights)
    iter_list = list(iter(i) for i in iter_seq)

    while iter_list:
        choice_i = random.choices(range(len(iter_list)), weights=weights, k=1)[0]
        try:
            yield next(iter_list[choice_i])
        except StopIteration:
            iter_list.pop(choice_i)
            weights.pop(choice_i)


def slice_into_sublists(lst: list[T]) -> Iterator[list[T]]:
    """
    >>> list(slice_into_sublists([1, 2, 3]))  # doctest: +NORMALIZE_WHITESPACE
    [[[1], [2], [3]],
     [[1], [2, 3]],
     [[1, 2], [3]],
     [[1, 2, 3]]]
    """
    for doslice in itertools.product([True, False], repeat=len(lst) - 1):
        slices = []
        start = 0
        for i, slicehere in enumerate(doslice, 1):
            if slicehere:
                slices.append(lst[start:i])
                start = i
        slices.append(lst[start:])
        yield slices
