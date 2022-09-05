import ast
import hashlib
import pickle
import pandas as pd
import os
import warnings

CACHE_BASE = os.getenv(
    "DUMB_COMPOSER_CACHE", os.path.join(os.getenv("HOME"), ".cache")
)

NO_CACHE = os.getenv("NO_CACHE", None)
HASHSEED = os.getenv("PYTHONHASHSEED", None)

HASHSEED_WARNED = False


def get_func_path(f):
    try:
        return f.__globals__["__spec__"].origin
    except AttributeError:
        # in __main__ __spec__ can be set to None
        return f.__globals__["__file__"]


def get_cache_dir(f, *args, cache_base=CACHE_BASE, **kwargs):
    _hash = lambda x: hashlib.sha256(x.encode()).hexdigest()
    # Below I separate args into those arguments that are file paths
    #   and those which are not, then I hash these separately. I'm
    #   not sure why I did that.
    paths, non_paths = [], []
    for arg in args:
        if isinstance(arg, str) and os.path.exists(arg):
            paths.append(arg)
        else:
            non_paths.append(str(arg))
    hashed_paths = [_hash(path) for path in paths]
    hashed_args = _hash(",".join(non_paths))
    kwargs = [f"{k}={v}" for k, v in kwargs.items()]
    hashed_kwargs = _hash(",".join(kwargs))
    cache_dir = os.path.join(
        cache_base, f.__name__, *hashed_paths, hashed_args, hashed_kwargs
    )
    return cache_dir


def get_f_hash_path(cache_dir):
    return os.path.join(cache_dir, "f_hash")


def get_cache_path(cache_dir):
    return os.path.join(cache_dir, "cache")


# hashing functions doesn't seem to work (the same function has different
# hashes across different runs in spite of PYTHONHASHSEED being set). Thus
# as a simpler solution we just check if the source file that contains the
# function has been changed.
# def finger_print(func):
#     # after https://stackoverflow.com/a/69216674/10155119
#     return str(hash(func.__code__.co_consts) + hash(func.__code__.co_code))


# def check_f_hash(f, cache_dir):
#     f_hash_path = get_f_hash_path(cache_dir)
#     if not os.path.exists(f_hash_path):
#         return False
#     with open(f_hash_path, "r") as inf:
#         f_hash = inf.read().strip()
#     print(f"comparing {finger_print(f)} and {f_hash}")
#     return finger_print(f) == f_hash


def check_f_hash(f, cache_dir):
    f_hash_path = get_f_hash_path(cache_dir)
    if not os.path.exists(f_hash_path):
        return False
    with open(f_hash_path, "r") as inf:
        f_hash = inf.read().strip()
    f_path = get_func_path(f)
    mtime = str(os.path.getmtime(f_path))
    return mtime == f_hash


def check_cache(cache_dir, f, *args):
    if not os.path.exists(cache_dir):
        return "CACHE_DOES_NOT_EXIST"
    if not check_f_hash(f, cache_dir):
        return "CACHE_DOES_NOT_EXIST"

    paths = [
        arg for arg in args if (isinstance(arg, str) and os.path.exists(arg))
    ]
    if paths:
        paths_mtime = max(os.path.getmtime(p) for p in paths)
        cache_mtime = os.path.getmtime(cache_dir)
        if cache_mtime <= paths_mtime:
            return "CACHE_DOES_NOT_EXIST"
    cache_path = get_cache_path(cache_dir)
    with open(cache_path, "rb") as inf:
        try:
            out = pickle.load(inf)
        except Exception as exc:
            print(f"Error loading cache: {exc}")
            return "CACHE_DOES_NOT_EXIST"
    return out


def write_cache(cache_dir, f, return_value):
    f_path = get_func_path(f)
    mtime = str(os.path.getmtime(f_path))
    with open(get_f_hash_path(cache_dir), "w") as outf:
        # print(f"storing {finger_print(f)}")
        # outf.write(finger_print(f))
        outf.write(mtime)

    with open(get_cache_path(cache_dir), "wb") as outf:
        pickle.dump(return_value, outf)


def cacher(write_cache_f=write_cache, cache_base=CACHE_BASE):
    """A decorator for caching the results of function calls.

    A dir path is created inside cache_base by joining the args and kwargs to
    the decorated function. If that dir exists, we check whether the files in
    it are newer than all paths in args. (We don't check kwargs for paths.)
    If so, we return the cache. Otherwise, we call the function (but cache
    the result for next time).

    write_cache_f could be overwritten for customization.

    Checks if the file that the decorated function is in has changed. If so,
    updates cache (regardless of whether function itself has changed; it's not
    smart enough to tell.)
    """

    def wrap(f):
        if HASHSEED is None or HASHSEED == "random":
            global HASHSEED_WARNED
            if not HASHSEED_WARNED:
                warnings.warn(
                    f"PYTHONHASHSEED is set to 'random' or is not "
                    "set; caching is disabled"
                )
                HASHSEED_WARNED = True
            return f

        if NO_CACHE is not None:
            warnings.warn(
                f"NO_CACHE is set, caching of {f.__name__} will be disabled"
            )
            return f

        def f1(*args, **kwargs):
            cache_dir = get_cache_dir(f, *args, cache_base=cache_base, **kwargs)
            cached = check_cache(cache_dir, f, *args)
            if cached != "CACHE_DOES_NOT_EXIST":
                return cached
            out = f(*args, **kwargs)
            os.makedirs(cache_dir, exist_ok=True)
            write_cache_f(cache_dir, f, out)
            return out

        return f1

    return wrap
