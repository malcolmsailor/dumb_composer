import shutil
import os
import subprocess
import pytest
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
        failed_files = []
        png_paths = []
        print("\nConverting midi files to pngs (ctrl-C to interrupt)")
        for f in tqdm(files):
            png_path = f"{f[:-4]}.png"
            try:
                subprocess.run(
                    f"mid2hum {f} | autobeam | verovio - -o - | convert - {png_path}",
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                failed_files.append(f)
            else:
                png_paths.append(png_path)
        if failed_files:
            print("Failed converting following files:")
            for file in failed_files:
                print(f"   {file}")
        print("New png files:")
        for f in png_paths:
            print("   ", f)

        memory_updater()
    except KeyboardInterrupt:
        pass


def pytest_addoption(parser):
    parser.addoption(
        "--quick", action="store_true", help="run 'quick' version of tests"
    )


@pytest.fixture(scope="session")
def quick(request):
    return request.config.option.quick


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    _mid_to_png()
