import random
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Type, TypeVar, get_type_hints

import yaml

from dumb_composer.pitch_utils.types import SettingsBase, TimeStamp


def get_attribute_type(dataclass_class: Type[SettingsBase], attr_name: str):
    # Get the type hints for the dataclass attributes
    type_hints = get_type_hints(dataclass_class)

    # Print attribute names and their expected types
    if attr_name not in type_hints:
        raise ValueError(f"{attr_name=} not among the fields of {dataclass_class=}")
    return type_hints[attr_name]


def get_random_val(expected_type: Type, min_val, max_val):
    if expected_type in (TimeStamp, Fraction, float):
        value = min_val + random.random() * (max_val - min_val)
        return expected_type(value)
    if expected_type is int:
        return random.randint(min_val, max_val)

    raise ValueError(f"{expected_type=} not supported by get_random_val()")


S = TypeVar("S", bound=SettingsBase)


def load_config_from_yaml_basic(
    settings_class: Type[S], yaml_path: str | Path | None
) -> S:
    if yaml_path is None:
        return settings_class()
    # No randomization
    with open(yaml_path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    return settings_class(**config_dict)


def load_config_from_yaml(settings_class: Type[S], yaml_path: str | Path | None) -> S:
    if yaml_path is None:
        return settings_class()
    # Load configuration from YAML file
    with open(yaml_path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    max_range_keys = {}
    min_range_keys = {}
    other_keys = {}

    for key, val in config_dict.items():
        if key.startswith("MAX_"):
            max_range_keys[key] = val
        elif key.startswith("MIN_"):
            min_range_keys[key] = val
        elif key.startswith("CHOICES_"):
            # TODO: (Malcolm 2023-08-11)
            raise NotImplementedError
        elif key.startswith("WEIGHTS_"):
            # TODO: (Malcolm 2023-08-11)
            raise NotImplementedError
        else:
            other_keys[key] = val

    for min_key in min_range_keys:
        base_key = min_key[4:]  # Remove "MIN_"
        if base_key in other_keys:
            raise ValueError(f"Found {min_key=} but also {base_key=}")
        max_key = "MAX_" + base_key
        if max_key not in max_range_keys:
            raise ValueError(f"Found {min_key=} but missing {max_key=}")

    for max_key, max_val in max_range_keys.items():
        base_key = max_key[4:]  # Remove "MAX_"
        if base_key in other_keys:
            raise ValueError(f"Found {max_key=} but also {base_key=}")
        min_key = "MIN_" + base_key
        if min_key not in min_range_keys:
            raise ValueError(f"Found {max_key=} but missing {min_key=}")

        min_val = min_range_keys[min_key]
        expected_type = get_attribute_type(settings_class, base_key)
        random_val = get_random_val(expected_type, min_val, max_val)
        other_keys[base_key] = random_val

    return settings_class(**other_keys)
