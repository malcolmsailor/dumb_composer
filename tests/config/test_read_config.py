import os
import random
import tempfile

import pytest
import yaml

from dumb_composer.config.read_config import load_config_from_yaml
from dumb_composer.structural_partitioner import StructuralPartitionerSettings

STRUCTURAL_PARTITIONER_YAML = """
never_split_dur_in_beats: 2.0
MAX_always_split_dur_in_bars: 4.0
MIN_always_split_dur_in_bars: 2.0
"""

SETTINGS_CLASS_AND_YAML = [(StructuralPartitionerSettings, STRUCTURAL_PARTITIONER_YAML)]


def read_yaml_helper(file_path):
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


@pytest.fixture(params=SETTINGS_CLASS_AND_YAML)
def settings_class_and_yaml_file_path(request):
    settings_class, yaml_contents = request.param
    # Create a temporary file and write YAML content to it
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(yaml_contents)
        temp_file_path = temp_file.name

    yield settings_class, temp_file_path

    # Clean up: delete the temporary file
    os.remove(temp_file_path)


def test_load_config_from_yaml(settings_class_and_yaml_file_path):
    random.seed(42)
    settings_class, yaml_file_path = settings_class_and_yaml_file_path
    settings = load_config_from_yaml(settings_class, yaml_file_path)
    dict = read_yaml_helper(yaml_file_path)
    for key, val in dict.items():
        if key.startswith("MAX_"):
            base_key = key[4:]
            assert getattr(settings, base_key) <= val
        elif key.startswith("MIN_"):
            base_key = key[4:]
            assert getattr(settings, base_key) >= val
        else:
            assert getattr(settings, key) == val
