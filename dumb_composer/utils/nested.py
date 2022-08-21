from typing import Callable, Sequence, Tuple, Union
import warnings

try:
    import numpy as np  # type:ignore

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False

try:
    import torch  # type:ignore

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

try:
    import pandas as pd  # type:ignore

    HAS_PANDAS = True
except ModuleNotFoundError:
    HAS_PANDAS = False


def nested(
    coerce_to_list: bool = False,
    types_to_process: Union[type, Tuple[type]] = None,
    # fail_silently: bool = False,
) -> Callable:
    """Decorator to extend a function to arbitrarily deep/nested list-likes or
    dicts.

    If the argument to the decorated function is a non-string sequence
    or a numpy array, the
    function will be recalled recursively on every item of the sequence.
    Otherwise, `func` will be called on the argument.

    Keyword args:
        - types_to_process: if passed, elements that match this type will be
            returned unchanged. This could, for instance, be used to process
            only the ints in a dict of form {str:int}.
    """
    if types_to_process is not None and isinstance(types_to_process, type):
        types_to_process = (types_to_process,)

    def _decorator(func: Callable) -> Callable:
        def f(item, *args, **kwargs):
            if isinstance(item, Sequence) and not isinstance(item, str):
                if coerce_to_list:
                    return list(
                        f(sub_item, *args, **kwargs) for sub_item in item
                    )
                return type(item)(
                    f(sub_item, *args, **kwargs) for sub_item in item
                )
            elif isinstance(item, dict):
                if coerce_to_list:
                    warnings.warn("Can't coerce dict to list")
                return {
                    f(k, *args, **kwargs): f(v, *args, **kwargs)
                    for k, v in item.items()
                }
            elif HAS_NUMPY and isinstance(item, np.ndarray):
                if coerce_to_list:
                    return list(
                        f(sub_item, *args, **kwargs) for sub_item in item
                    )
                return np.fromiter(
                    (f(sub_item, *args, **kwargs) for sub_item in item),
                    dtype=item.dtype,
                )
            elif HAS_TORCH and isinstance(item, torch.Tensor):
                if not item.dim():
                    return func(item.item(), *args, **kwargs)
                if coerce_to_list:
                    return list(
                        f(sub_item, *args, **kwargs) for sub_item in item
                    )
                # e.g., "float32" if item.dtype is "torch.float32"
                base_dtype = item.dtype.__repr__().split(".")[1]
                return type(item)(
                    np.fromiter(
                        (f(sub_item, *args, **kwargs) for sub_item in item),
                        dtype=getattr(np, base_dtype),
                    )
                )
            elif HAS_PANDAS and isinstance(item, pd.Series):
                if coerce_to_list:
                    return list(
                        f(sub_item, *args, **kwargs) for sub_item in item
                    )
                return item.apply(f, args=args, **kwargs)
            else:
                if types_to_process is not None and not isinstance(
                    item, types_to_process
                ):
                    return item
                return func(item, *args, **kwargs)

        return f

    return _decorator
