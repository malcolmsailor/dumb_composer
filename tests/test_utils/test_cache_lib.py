import subprocess
import os
import shutil
import time
import tempfile
from pathlib import Path

from dumb_composer.utils.cache_lib import cacher, get_func_path


def test_get_func_path():
    def f():
        pass

    assert get_func_path(f) == os.path.realpath(__file__)
    assert os.path.samefile(
        get_func_path(cacher),
        os.path.join(
            os.path.realpath(os.path.dirname(__file__)),
            "..",
            "..",
            "dumb_composer",
            "utils",
            "cache_lib.py",
        ),
    )


def _make_temp_file(path):
    f_contents = str(time.time())
    with open(path, "w") as outf:
        outf.write(f_contents)


def test_cacher():
    temp_dir = tempfile.mkdtemp()
    _, path1 = tempfile.mkstemp()
    _, path2 = tempfile.mkstemp()

    try:
        f_execution_times = {}

        @cacher(cache_base=temp_dir)
        def f(path):
            f_execution_times[path] = time.time()
            with open(path, "r") as inf:
                return inf.read()

        _make_temp_file(path1)
        _make_temp_file(path2)
        f1_contents = f(path1)
        f2_contents = f(path2)

        # f should execute for path2 as well
        assert path2 in f_execution_times
        assert f_execution_times[path1] != f_execution_times[path2]

        f_last_ran_for_path1 = f_execution_times[path1]
        f_last_ran_for_path2 = f_execution_times[path2]

        # file 1 has not changed, f should not execute
        f1_contents_again = f(path1)
        assert f_execution_times[path1] == f_last_ran_for_path1
        assert f1_contents_again == f1_contents

        # touch file 2, f should execute
        Path(path2).touch()
        touched_f2_contents = f(path2)
        assert f_execution_times[path2] != f_last_ran_for_path2
        assert touched_f2_contents == f2_contents

        _make_temp_file(path1)
        changed_f1_contents = f(path1)
        # file 2 has changed, f should execute
        assert f_execution_times[path1] != f_last_ran_for_path1
        assert changed_f1_contents != f1_contents

        # # redefine f without changing it
        # @cacher(cache_base=temp_dir)
        # def f(path):
        #     f_execution_times[path] = time.time()
        #     with open(path, "r") as inf:
        #         return inf.read()

        # # f has not changed, contents should be same
        # redefined_f1_contents = f(path1)
        # assert f_execution_times[path1] == f_last_ran_for_path1
        # assert redefined_f1_contents == f1_contents

        # redefine f and change it
        @cacher(cache_base=temp_dir)
        def f(path):
            pointless_statement = None
            f_execution_times[path] = time.time()
            with open(path, "r") as inf:
                return inf.read()

        changed_f_f1_contents = f(path1)
        assert f_execution_times[path1] != f_last_ran_for_path1
        assert changed_f_f1_contents != f1_contents
    finally:
        print("removing temporary files")
        os.remove(path1)
        os.remove(path2)
        shutil.rmtree(temp_dir)


def test_cacher_across_runs():
    temp_dir = tempfile.mkdtemp()
    _, path = tempfile.mkstemp()
    try:

        def _bool_from_out(subprocess_out):
            contents = (
                subprocess_out.stdout.decode().strip().rsplit("\n", maxsplit=1)
            )
            if len(contents) == 2:
                print(contents[0])
                bool_str = contents[1]
            else:
                bool_str = contents[0]
            if not bool_str in ("True", "False"):
                raise ValueError()
            return eval(bool_str)

        Path(path).touch()
        helper_script = os.path.join(
            os.path.dirname((os.path.realpath(__file__))), "cache_helper.py"
        )
        result = _bool_from_out(
            subprocess.run(
                ["python3", helper_script, temp_dir, path],
                capture_output=True,
                check=True,
            )
        )
        assert result
        result = _bool_from_out(
            subprocess.run(
                ["python3", helper_script, temp_dir, path],
                capture_output=True,
                check=True,
            )
        )
        assert not result
        Path(helper_script).touch()
        result = _bool_from_out(
            subprocess.run(
                ["python3", helper_script, temp_dir, path],
                capture_output=True,
                check=True,
            )
        )
        assert result
    finally:
        print("removing temporary files")
        os.remove(path)
        shutil.rmtree(temp_dir)
