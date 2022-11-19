import logging
import os
import random
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

import albumentations as A


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = get_device()
logger = get_logger("main.log")
set_seed(42)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(
        self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1
    ):
        super().__init__()

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2,
            time_stripes_num=2,
            freq_drop_width=8 // 2,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input_data):
        x = input_data  # (batch_size, 3, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = x.transpose(2, 3)

        x = self.encoder(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "clipwise_output": clipwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
        }

        return output_dict


mean = (0.485, 0.456, 0.406)  # RGB
std = (0.229, 0.224, 0.225)  # RGB

albu_transforms = {
    "train": A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Cutout(max_h_size=5, max_w_size=16),
                    A.CoarseDropout(max_holes=4),
                ],
                p=0.5,
            ),
            A.Normalize(mean, std),
        ]
    ),
    "valid": A.Compose(
        [
            A.Normalize(mean, std),
        ]
    ),
}


class CFG:
    EXP_ID = "018"

    ######################
    # Globals #
    ######################
    seed = 42
    epochs = 5
    train = True
    folds = [0, 1, 2, 3, 4]
    img_size = 224
    main_metric = "epoch_f1_at_03"
    minimize_metric = False

    ######################
    # Dataset #
    ######################
    transforms = {"train": [{"name": "Normalize"}], "valid": [{"name": "Normalize"}]}
    period = 5
    n_mels = 224
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {"n_mels": 224, "fmin": 20, "fmax": 16000}

    target_columns = "afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                      barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                      brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                      cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                      comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                      fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                      hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                      jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                      madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                      norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                      reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                      saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov".split()

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {"batch_size": 16, "num_workers": 0, "shuffle": True},
        "valid": {"batch_size": 32, "num_workers": 0, "shuffle": False},
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b0_ns"
    pooling = "max"
    pretrained = False
    num_classes = 152
    in_channels = 3

    N_FOLDS = 5
    LR = 1e-3
    T_max = 10
    min_lr = 1e-6


AUDIO_PATH = "data/raw/train_audio"
CLASSES = sorted(os.listdir(AUDIO_PATH))
NUM_CLASSES = len(CLASSES)


class AudioParams:
    """
    Parameters used for the audio data
    """

    sr = 32000
    duration = 5
    # Melspectrogram
    n_mels = 224
    fmin = 20
    fmax = 16000


def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=params.sr,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray):
        self.df = df
        # self.clip = clip
        self.clip = np.concatenate([clip, clip, clip])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        # end_index = int(SR * (end_seconds + (self.train_period - 5) / 2) + len(self.clip) // 3)
        # start_index = int(SR * (start_seconds - (self.train_period - 5) / 2) + len(self.clip) // 3)

        # y = self.clip[start_index:end_index].astype(np.float32)
        image = self.clip[SR * start_seconds : SR * end_seconds].astype(np.float32)
        image = np.nan_to_num(image)

        image = compute_melspec(image, AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = albu_transforms["valid"](image=image)["image"].T

        return {
            "image": image,
            "row_id": row_id,
        }


model_paths = ["fold-0.bin"]

models = []
for p in model_paths:
    model = TimmSED(
        base_model_name=CFG.base_model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels,
    )

    model.to(device)
    model.load_state_dict(torch.load(p))
    model.eval()
    models.append(model)

print()

TARGET_SR = 32000
DATADIR = Path("data/raw/test_soundscapes/")

all_audios = list(DATADIR.glob("*.ogg"))
sample_submission = pd.read_csv("data/raw/sample_submission.csv")
sample_submission


def prediction_for_clip(
    test_df: pd.DataFrame, clip: np.ndarray, models, threshold=0.05, threshold_long=None
):

    dataset = TestDataset(
        df=test_df,
        clip=clip,
    )
    loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     [model.eval() for model in models]
    prediction_dict = {}
    for data in tqdm(loader):
        row_id = data["row_id"][0]
        image = data["image"].to(device)

        with torch.no_grad():
            probas = []
            probas_long = []
            for model in models:
                with torch.cuda.amp.autocast():
                    output = model(image)
                probas.append(
                    output["clipwise_output"].detach().cpu().numpy().reshape(-1)
                )
                # probas_long.append(clipwise_pred_long.detach().cpu().numpy().reshape(-1))
            probas = np.array(probas)
            # probas_long = np.array(probas_long)
        #             probas = np.array([model(image)[1].detach().cpu().numpy().reshape(-1) for model in models])
        if threshold_long is None:
            events = probas.mean(0) >= threshold
        else:
            events = (
                (probas.mean(0) >= threshold).astype(int)
                + (probas_long.mean(0) >= threshold_long).astype(int)
            ) >= 2
        labels = np.argwhere(events).reshape(-1).tolist()
        #         labels = labels[:2]
        if len(labels) == 0:
            prediction_dict[str(row_id)] = "nocall"
        else:
            labels_str_list = list(map(lambda x: CFG.target_columns[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[str(row_id)] = label_string
    return prediction_dict


def prediction(test_audios, threshold=0.05, threshold_long=None):

    # models = [model]
    warnings.filterwarnings("ignore")
    prediction_dicts = {}
    for audio_path in test_audios:
        with timer(f"Loading {str(audio_path)}", logger):
            clip, _ = sf.read(audio_path, always_2d=True)
            clip = np.mean(clip, 1)

        seconds = []
        row_ids = []
        for second in range(5, 65, 5):
            row_id = "_".join(audio_path.name.split(".")[:-1]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)
        print(row_ids)
        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        with timer(f"Prediction on {audio_path}", logger):
            prediction_dict = prediction_for_clip(
                test_df,
                clip=clip,
                models=models,
                threshold=threshold,
                threshold_long=threshold_long,
            )
        #         row_id = list(prediction_dict.keys())
        #         birds = list(prediction_dict.values())
        #         prediction_df = pd.DataFrame({
        #             "row_id": row_id,
        #             "birds": birds
        #         })
        #         prediction_dfs.append(prediction_df)
        #     prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
        prediction_dicts.update(prediction_dict)
    return prediction_dicts


threshold = 0.05
threshold_long = None  # 0.05

prediction_dicts = prediction(
    test_audios=all_audios, threshold=threshold, threshold_long=threshold_long
)
print(prediction_dicts)

for i in range(len(sample_submission)):
    sample = sample_submission.row_id[i]
    key = sample.split("_")[0] + "_" + sample.split("_")[1] + "_" + sample.split("_")[3]
    target_bird = sample.split("_")[2]
    print(key, target_bird)
    if key in prediction_dicts:
        sample_submission.iat[i, 1] = target_bird in prediction_dicts[key]
sample_submission.to_csv("submission.csv", index=False)
