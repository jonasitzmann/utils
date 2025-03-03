import inspect
import logging
import os
import pstats
import re
from contextlib import contextmanager
from cProfile import Profile
from datetime import datetime
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd

_SUBDIR = ""
_OPTIONS: dict = dict()
_MAX_CALLS: Final[int] = 99
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def set_subdir(subdir):
    global _SUBDIR
    _SUBDIR = subdir


def set_options(**new_options):
    _OPTIONS.update(new_options)


def format_number_leading_zeros(i):
    num_digits = len(str(_MAX_CALLS))
    return f"{i:0{num_digits}d}"


def get_out_dir(profiling_dir: Optional[Path | str] = None, filename: Optional[str] = None):
    default_out_dir = Path.home() / "profiling"
    parent = Path(profiling_dir or str(os.environ.get("PROFILING_DIR", default_out_dir))) / _SUBDIR
    if _OPTIONS:
        parent /= "__".join(f"{k}={v}" for k, v in _OPTIONS.items())
    for i in range(1, _MAX_CALLS + 1):
        dirname = f"{filename}_{format_number_leading_zeros(i)}"
        out_dir = parent / dirname
        if not out_dir.is_dir():
            os.makedirs(out_dir, exist_ok=True)
            set_subdir("")
            return out_dir
    raise RuntimeError(f"reached maximum number {_MAX_CALLS} of profiling files")


def profile_cls(print_stats=False, funcs=None, profiling_dir: Optional[Path | str] = None):
    """
    funcs: List of functions to profile. Default: all functions (which may lead to nested profiling, see profile_fn()
    if used in combination with attrs, put this decorator below @attrs:
    @attrs(...)
    @profile_cls(...)
    class Foo: ...

    To profile external classes:
    from foolib import Foo
    FooWithProfiling = profile_cls(funcs=[Foo.bar], print_stats=True)(Foo)
    foo_instance = FooWithProfiling()
    """

    def decorate_cls(cls):
        func_list = funcs or [func for f_name, func in inspect.getmembers(cls, inspect.isfunction)]
        for func in func_list:
            logger.info(f"decorating {func.__qualname__}")
            setattr(cls, func.__name__, profile_fn(print_stats=print_stats, profiling_dir=profiling_dir)(func))
        return cls

    return decorate_cls


def profile_fn(print_stats=False, profiling_dir: Optional[Path | str] = None):
    """
    When most of the time is spent in "{method 'enable' of '_lsProf.Profiler' objects}", there is most likely nested
    profiling, which is not supported. If a() calls b(), only decorate a(), not b()
    """

    def profile_no_args(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_file = Path(inspect.getsourcefile(func) or "").name
            filename = f"{func_file}__{func.__qualname__}"
            with profile(print_stats=print_stats, profiling_dir=profiling_dir, filename=filename):
                return func(*args, **kwargs)

        return wrapper

    return profile_no_args


@contextmanager
def profile(print_stats=False, profiling_dir: Optional[Path | str] = None, filename="profiling_results"):
    """
    Known issue with cPython: (some) numpy functions are not considered in the output.
    Wrapping these in python functions solves this.
    See functions `sqrt` and `square` in this file as examples.
    """
    t0 = datetime.now()
    with Profile() as prof:
        yield
    t1 = datetime.now()
    logger.info(f"time passed: {t1 - t0}")
    out_dir = get_out_dir(profiling_dir, filename)
    profiling_results_path = out_dir / f"{filename}.prof"
    txt_path = profiling_results_path.with_suffix(".txt")
    prof.dump_stats(profiling_results_path)
    string_stream = StringIO()
    pstats.Stats(prof, stream=string_stream).strip_dirs().sort_stats("cumulative").print_stats()
    profiling_results = string_stream.getvalue()
    if print_stats:
        logger.info(profiling_results)
    with open(txt_path, "w") as file_out_stream:
        file_out_stream.write(profiling_results)
    lines = profiling_results.split("\n")[4:]
    lines = ["\t".join(re.split("\s+", l.strip(), maxsplit=5)) for l in lines]
    tab_separated_results = "\n".join(lines)
    df = pd.read_csv(StringIO(tab_separated_results), sep="\t")
    df.to_csv(txt_path.with_suffix(".csv"), index=False)


@profile_fn(print_stats=True)
def do_something():
    x = get_random_array()
    x_squared = square(x)
    x_sqrt = sqrt(x)


def get_random_array():
    return np.random.random([100_000_000])


def square(x):
    return x * x


def sqrt(x):
    return np.sqrt(x)


def main():
    do_something()


# @profile_cls()
# class Worker:
#     def fn_1(self):
#         do_something()
#
#     def fn_2(self):
#         get_random_array()
#

if __name__ == "__main__":
    main()
    worker = Worker()
    worker.fn_1()
    worker.fn_2()
