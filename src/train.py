import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from utils import set_seed, AverageMeter, MetricMeter

from tqdm import tqdm

import transformers
from torch.cuda.amp import autocast, GradScaler
import glob
import ast

from modeling.TimmSED import TimmSED
from data.datasets import WaveformDataset

all_path = glob.glob("data/interim/*/*.npy")
train = pd.read_csv("data/raw/train_metadata.csv")
train["new_target"] = (
    train["primary_label"]
    + " "
    + train["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))
)
train["len_new_target"] = train["new_target"].map(lambda x: len(x.split()))
train["filename"] = train["filename"].map(lambda x: x.replace("/", os.sep))

path_df = pd.DataFrame(all_path, columns=["file_path"])
path_df["filename"] = path_df["file_path"].map(
    lambda x: x.split(os.sep)[-2] + os.sep + x.split(os.sep)[-1][:-4]
)

train = pd.merge(train, path_df, on="filename")

Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train["primary_label"])):
    train.loc[val_index, "kfold"] = int(n)
train["kfold"] = train["kfold"].astype(int)
train.to_csv("train_folds.csv", index=False)


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = "EX005"
    seed = 71
    epochs = 23
    cutmix_and_mixup_epochs = 18
    folds = [0]  # [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    train_bs = 16  # 32
    valid_bs = 32  # 64
    base_model_name = "tf_efficientnet_b0_ns"
    EARLY_STOPPING = True
    DEBUG = False  # True
    EVALUATION = "AUC"
    apex = True

    pooling = "max"
    pretrained = True
    num_classes = 152
    in_channels = 3
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

    img_size = 224  # 128
    main_metric = "epoch_f1_at_03"

    period = 5
    n_mels = 224  # 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {"n_mels": 224, "fmin": 20, "fmax": 16000}  # 128,


OUTPUT_DIR = f"./"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_loss(y_true, y_pred):
    return metrics.roc_auc_score(np.array(y_true), np.array(y_pred))


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def loss_fn(logits, targets):
    loss_fct = BCEFocal2WayLoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data["image"].to(device)
        targets = data["targets"].to(device)
        with autocast(enabled=CFG.apex):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def train_mixup_cutmix_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data["image"].to(device)
        targets = data["targets"].to(device)

        if np.random.rand() < 0.5:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = mixup_criterion(outputs, new_targets)
        else:
            inputs, new_targets = cutmix(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = cutmix_criterion(outputs, new_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(new_targets[0], outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data["image"].to(device)
            targets = data["targets"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def inference_fn(model, data_loader, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    final_output = []
    final_target = []
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            inputs = data["image"].to(device)
            targets = data["targets"].to(device).detach().cpu().numpy().tolist()
            output = model(inputs)
            output = output["clipwise_output"].cpu().detach().cpu().numpy().tolist()
            final_output.extend(output)
            final_target.extend(targets)
    return final_output, final_target


def calc_cv(model_paths):
    df = pd.read_csv("train_folds.csv")
    y_true = []
    y_pred = []
    for fold, model_path in enumerate(model_paths):
        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels,
        )

        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        val_df = df[df.kfold == fold].reset_index(drop=True)
        dataset = WaveformDataset(df=val_df, mode="valid")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=CFG.valid_bs,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        final_output, final_target = inference_fn(model, dataloader, device)
        y_pred.extend(final_output)
        y_true.extend(final_target)
        torch.cuda.empty_cache()

        f1_03 = metrics.f1_score(
            np.array(y_true), np.array(y_pred) > 0.3, average="micro"
        )
        print(f"micro f1_0.3 {f1_03}")

    f1_03 = metrics.f1_score(np.array(y_true), np.array(y_pred) > 0.3, average="micro")
    f1_05 = metrics.f1_score(np.array(y_true), np.array(y_pred) > 0.5, average="micro")

    print(f"overall micro f1_0.3 {f1_03}")
    print(f"overall micro f1_0.5 {f1_05}")
    return


# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    print("=" * 100)
    print(f"Fold {fold} Training")
    print("=" * 100)

    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)

    train_dataset = WaveformDataset(df=trn_df, mode="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.train_bs,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

    valid_dataset = WaveformDataset(df=val_df, mode="valid")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.valid_bs,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    model = TimmSED(
        base_model_name=CFG.base_model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels,
    )

    optimizer = transformers.AdamW(
        model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=CFG.ETA_MIN, T_max=500
    )

    model = model.to(device)

    min_loss = 999
    best_score = -np.inf

    for epoch in range(CFG.epochs):
        print("Starting {} epoch...".format(epoch + 1))

        start_time = time.time()

        if epoch < CFG.cutmix_and_mixup_epochs:
            train_avg, train_loss = train_mixup_cutmix_fn(
                model, train_dataloader, device, optimizer, scheduler
            )
        else:
            train_avg, train_loss = train_fn(
                model, train_dataloader, device, optimizer, scheduler
            )

        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s"
        )
        print(
            f"Epoch {epoch+1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}"
        )
        print(
            f"Epoch {epoch+1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}"
        )

        if valid_avg["f1_at_03"] > best_score:
            print(
                f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_03']}"
            )
            print(
                f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}"
            )
            torch.save(model.state_dict(), f"fold-{fold}.bin")
            best_score = valid_avg["f1_at_03"]


model_paths = [f"fold-{i}.bin" for i in CFG.folds]

calc_cv(model_paths)
