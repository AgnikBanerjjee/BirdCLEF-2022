import os
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

SR = 32000
USE_SEC = 30
AUDIO_PATH = "data/raw/train_audio"
NUM_WORKERS = 4
CLASSES = sorted(os.listdir(AUDIO_PATH))


def audio_to_array(path):
    y, sr = sf.read(path, always_2d=True)
    y = np.mean(y, 1)  # stereo to mono

    # remove first and last seconds
    if len(y) > SR:
        y = y[SR:-SR]

    # keep only first x seconds
    if len(y) > SR * USE_SEC:
        y = y[: SR * USE_SEC]
    return y


def save_(path):
    save_path = "data/interim/" + "/".join(path.split("/")[-2:])
    np.save(save_path, audio_to_array(path))


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train_metadata.csv")
    train["file_path"] = AUDIO_PATH + "/" + train["filename"]
    paths = train["file_path"].values

    for dir_ in CLASSES:
        _ = os.makedirs("data/interim/" + dir_, exist_ok=True)
    _ = Parallel(n_jobs=NUM_WORKERS)(
        delayed(save_)(AUDIO_PATH) for AUDIO_PATH in tqdm(paths)
    )
