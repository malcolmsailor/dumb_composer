import shutil
import os
import subprocess
from tempfile import mkstemp
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
    # basenames = [f for f in os.listdir(TEST_OUT_DIR) if f.endswith(".mid")]
    basenames = [
        os.path.join(dirname[len(TEST_OUT_DIR) :].lstrip(os.path.sep), f)
        for (dirname, _, fnames) in os.walk(TEST_OUT_DIR)
        for f in fnames
        if f.endswith(".mid")
    ]
    files, memory_updater = get_changed_files("mids", basenames, TEST_OUT_DIR)

    if not files:
        return
    try:
        failed_files = {}
        png_paths = []
        print("\nConverting midi files to pngs (ctrl-C to interrupt)")
        for f in tqdm(files):
            png_path = f"{f[:-4]}.png"
            _, temp_path = mkstemp(suffix=".xml")
            try:
                # mid2hum doesn't seem to handle time signatures correctly so
                #   we use intermediate xml conversion via mscore
                subprocess.run(
                    # f"mid2hum {f} | autobeam | verovio - -o - | convert - {png_path}",
                    f"mscore {f} -o {temp_path}",
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    f"musicxml2hum {temp_path} | autobeam | verovio - -o - | convert - {png_path}",
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                intermediate_files = []
                intermediate_dir = os.path.join(
                    TEST_OUT_DIR, "intermediate_files"
                )
                os.makedirs(intermediate_dir, exist_ok=True)
                xml_path = os.path.join(
                    intermediate_dir,
                    os.path.basename(os.path.splitext(f)[0] + ".xml"),
                )
                try:
                    subprocess.run(
                        f"mscore {f} -o {xml_path}",
                        shell=True,
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    pass
                else:
                    intermediate_files.append(xml_path)
                    krn_path = os.path.join(
                        intermediate_dir,
                        os.path.basename(os.path.splitext(f)[0] + ".krn"),
                    )
                    try:
                        subprocess.run(
                            f"musicxml2hum {xml_path} > {krn_path}",
                            shell=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError:
                        pass
                    else:
                        intermediate_files.append(krn_path)
                failed_files[f] = intermediate_files
            else:
                png_paths.append(png_path)
            finally:
                os.remove(temp_path)
        if failed_files:
            print("Failed converting following files:")
            for f, intermediate_files in failed_files.items():
                print(f"   {f}")
                if intermediate_files:
                    print("   Intermediate files:")
                    for ff in intermediate_files:
                        print(f"        {ff}")
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
    parser.addoption(
        "--romantext", type=str, help="path to romantext file", default=""
    )


@pytest.fixture(scope="session")
def quick(request):
    return request.config.option.quick


@pytest.fixture(scope="session")
def romantext(request):
    return request.config.option.romantext


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    _mid_to_png()
