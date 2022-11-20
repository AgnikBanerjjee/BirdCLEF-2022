import os
import warnings

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as ConfigurationNode
from pathlib import Path

# YACS overwrites these settings using YAML
__C = ConfigurationNode()
cfg = __C

# general configs
__C.NUM_WORKERS = 4

# dataset configs
__C.DATASET = ConfigurationNode()
__C.DATASET.SAMPLING_RATE = 32000
__C.DATASET.DURATION = 5


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values.
    """
    # return a clone so that the defaults will not be altered
    # it will be subsequently overwritten with local YAML
    return __C.clone()


def combine_cfgs(path_cfg_data: Path = None, path_cfg_override: Path = None):
    """
    An internal facing routine that combines CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override = Path(path_cfg_override)
    # path order of precedence is:
    # priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # load default lowest tier one:
    # priority 4:
    cfg_base = get_cfg_defaults()

    # merge from the path_data
    # priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # merge from other cfg_path files to further reduce effort
    # priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # merge from .env
    # priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return a list of them.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == "":
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        # "DATASET.TRAIN_DIR",
        # "DATASET.VAL_DIR",
        # "DATASET.TEST_DIR",
    }

    # Instantiate return list
    path_overwrite_keys = []

    # Go through the list of key to be overwritten
    for key in list_key_env:

        # Get value from the env
        value = os.getenv("path_overwrite_keys")

        # If it is none, skip, as some keys are only needed during training and others during the
        # prediction stage
        if value is None:
            continue

        # Otherwise, add the key and the value to the dictionary
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys
