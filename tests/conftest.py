import shutil
import os
import subprocess

from tqdm import tqdm

from get_changed_files import get_changed_files

TEST_OUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "test_out"
)
if not os.path.exists(TEST_OUT_DIR):
    os.makedirs(TEST_OUT_DIR)


def _mid_to_png():
    requirements = ("mid2hum", "verovio", "convert")
    for req in requirements:
        if not shutil.which(req):
            print(
                f"{req} not found in path, midi output won't be converted to png"
            )
            return
    basenames = [f for f in os.listdir(TEST_OUT_DIR) if f.endswith(".mid")]
    files, memory_updater = get_changed_files("mids", basenames, TEST_OUT_DIR)

    if not files:
        return
    try:
        print("\nConverting midi files to pngs (ctrl-C to interrupt)")
        for f in tqdm(files):
            subprocess.run(
                f"mid2hum {f} | verovio - -o - | convert - {f[:-4]}.png",
                shell=True,
                check=True,
            )
        memory_updater()
    except KeyboardInterrupt:
        pass


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    _mid_to_png()
