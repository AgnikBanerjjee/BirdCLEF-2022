"""
make dataset
Usage:
    make_dataset.py [--cfg=<path>]
    make_dataset.py -h | --help
Options:
    -h --help           show this screen help
    --cfg=<path>        config path 
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm
from pathlib import Path, PurePath
from joblib import Parallel, delayed

from docopt import docopt
from dotenv import find_dotenv, load_dotenv

from src.config import get_cfg_defaults

load_dotenv(find_dotenv())
DIR_RAW = Path(os.getenv("DIR_RAW"))
DIR_PROCESSED = Path(os.getenv("DIR_PROCESSED"))

from src.config import get_cfg_defaults


def audio_to_array(path, cfg):
    sr = cfg.DATASET.SAMPLING_RATE
    duration = cfg.DATASET.DURATION

    y, sr = sf.read(path, always_2d=True)
    
    # convert stereo to mono
    y = np.mean(y, 1)  

    # remove first and last seconds
    if len(y) > sr:
        y = y[sr:-sr]

    # keep only first x seconds
    if len(y) > sr * duration:
        y = y[: sr * duration]
    return y


def save_(path, cfg):
    save_path = DIR_PROCESSED / os.path.join(*str(path).split(os.sep)[-2:])
    np.save(save_path, audio_to_array(path, cfg))


def make_dataset(cfg):
    train = pd.read_csv(DIR_RAW / "train_metadata.csv")
    train["file_path"] = DIR_RAW / "train_audio" / train["filename"]
    paths = train["file_path"].values

    classes = sorted(os.listdir(DIR_RAW / "train_audio"))
    for dir_ in classes:
        _ = os.makedirs(DIR_PROCESSED / dir_, exist_ok=True)

    _ = Parallel(n_jobs=cfg.NUM_WORKERS)(
        delayed(save_)(path, cfg) for path in tqdm(paths)
    )


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    cfg_path = arguments["--cfg"]
    cfg = get_cfg_defaults()
    if cfg_path is not None:
        cfg.merge_from_file(cfg_path)
    make_dataset(cfg)
