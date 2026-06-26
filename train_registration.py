import os
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
from baseline_model import BaselineRegistrationModel
from tlrn_model import TLRNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_DIR  = ".../train"
TEST_DIR   = ".../test"
IMG_SIZE   = (128, 128)
CROP_DIST  = 25
BATCH_SIZE = 1
EPOCHS     = 100
LR         = 1e-4
SIM_WEIGHT = 1.0
REG_WEIGHT = 0.1
SAVE_DIR   = "./saved_models_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)

log_path = os.path.join(SAVE_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import SimpleITK as sitk
from collections import Counter

class CardiacDataset(data.Dataset):
    def __init__(self, root_dir, img_size=IMG_SIZE, crop_dist=CROP_DIST):
        self.img_size   = img_size
        self.crop_dist  = crop_dist
        self.samples    = []

        for label in ['0', '1']:
            folder = os.path.join(root_dir, label)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.mhd'):
                    fpath = os.path.join(folder, fname)
                    arr   = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
                    if arr.shape[0] >= 24:
                        self.samples.append(fpath)

        print(f"  Samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(self.samples[idx])).astype(np.float32)

        arr = arr[:24]
        arr = arr[:, self.crop_dist:-self.crop_dist,
                     self.crop_dist:-self.crop_dist]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        vol_t = torch.tensor(arr).unsqueeze(1)
        vol_t = F.interpolate(vol_t, size=self.img_size,
                              mode="bilinear", align_corners=False)
        return vol_t


def get_dataloaders(train_dir, val_split=0.2, seed=42):
    dataset  = CardiacDataset(train_dir)
    n_val    = int(len(dataset) * val_split)
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    print(f"  Train: {n_train}  Val: {n_val}")

    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader


mse_loss = nn.MSELoss()

def l1_smoothness_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return (torch.mean(dx) + torch.mean(dy)) / 2.0


def run_epoch(net, loader, model_name, optimizer=None, train=True):
    net.train() if train else net.eval()
    total_sim, total_reg, total_all = 0., 0., 0.
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames in loader:
            frames = frames.to(device)
            T      = frames.shape[1]
            source = frames[:, 0]

            if model_name == "baseline":
                sim_loss_seq, reg_loss_seq = 0., 0.
                for t in range(1, T):
                    target = frames[:, t]
                    warped, flow, _ = net(source, target)
                    sim_loss_seq += mse_loss(warped, target)
                    reg_loss_seq += l1_smoothness_loss(flow)
                sim_loss_seq /= (T - 1)
                reg_loss_seq /= (T - 1)

            elif model_name == "tlrn":
                warped_list, flow_list, _ = net(frames)
                sim_loss_seq = sum(
                    mse_loss(warped_list[i], frames[:, i + 1])
                    for i in range(len(warped_list))
                ) / len(warped_list)
                reg_loss_seq = sum(
                    l1_smoothness_loss(flow_list[i])
                    for i in range(len(flow_list))
                ) / len(flow_list)

            loss = SIM_WEIGHT * sim_loss_seq + REG_WEIGHT * reg_loss_seq

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_sim += sim_loss_seq.item()
            total_reg += reg_loss_seq.item()
            total_all += loss.item()
            n_batches += 1

    return total_all / n_batches, total_sim / n_batches, total_reg / n_batches


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "tlrn"],
                        help="Which model to train: baseline or tlrn")
    args = parser.parse_args()

    MODEL_NAME = args.model
    SAVE_DIR   = f"./saved_models_{MODEL_NAME}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    log_path = os.path.join(SAVE_DIR, "training.log")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    train_loader, val_loader = get_dataloaders(TRAIN_DIR)

    sample  = next(iter(train_loader))
    inshape = tuple(sample.shape[-2:])
    print(f"\nModel     : {MODEL_NAME}")
    print(f"Input shape: {inshape}")

    model_kwargs = dict(
        inshape=inshape,
        nb_unet_features=[[16, 32], [32, 32, 16, 16]],
        nb_unet_conv_per_level=1,
        int_steps=7,
        int_downsize=2,
        src_feats=1,
        trg_feats=1,
        unet_half_res=True,
    )

    if MODEL_NAME == "baseline":
        net = BaselineRegistrationModel(**model_kwargs).to(device)
    elif MODEL_NAME == "tlrn":
        net = TLRNModel(**model_kwargs).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_total": [], "train_sim": [], "train_reg": [],
               "val_total":   [], "val_sim":   [], "val_reg":   []}

    best_val_loss  = float("inf")
    from datetime import datetime
    run_timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info(f"Training started — model: {MODEL_NAME}")
    logger.info(f"Epochs: {EPOCHS} | LR: {LR} | SIM_W: {SIM_WEIGHT} | REG_W: {REG_WEIGHT}")
    logger.info(f"Save dir: {SAVE_DIR}")
    logger.info("=" * 70)
    logger.info(f"{'Epoch':>6} | {'Tr Total':>9} {'Tr Sim':>9} {'Tr Reg':>9} | "
                f"{'Vl Total':>9} {'Vl Sim':>9} {'Vl Reg':>9} | {'Best':>5}")

    print("\nStarting training...\n")
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):

        tr_total, tr_sim, tr_reg = run_epoch(net, train_loader, MODEL_NAME, optimizer, train=True)
        vl_total, vl_sim, vl_reg = run_epoch(net, val_loader,   MODEL_NAME, optimizer=None, train=False)
        scheduler.step()

        history["train_total"].append(tr_total)
        history["train_sim"].append(tr_sim)
        history["train_reg"].append(tr_reg)
        history["val_total"].append(vl_total)
        history["val_sim"].append(vl_sim)
        history["val_reg"].append(vl_reg)

        is_best = vl_total < best_val_loss
        if is_best:
            best_val_loss = vl_total
            best_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best_{run_timestamp}.pth")
            torch.save(net.state_dict(), best_path)

        logger.info(
            f"{epoch+1:>6} | {tr_total:>9.4f} {tr_sim:>9.4f} {tr_reg:>9.4f} | "
            f"{vl_total:>9.4f} {vl_sim:>9.4f} {vl_reg:>9.4f} | "
            f"{'*' if is_best else ''}"
        )

        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(SAVE_DIR, f"{MODEL_NAME}_checkpoint_epoch{epoch+1}_{run_timestamp}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt)
            tqdm.write(f"  Checkpoint saved → {ckpt}")

    final_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final_{run_timestamp}.pth")
    torch.save(net.state_dict(), final_path)
    logger.info("=" * 70)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model  → {best_path}")
    logger.info(f"Final model → {final_path}")
    logger.info(f"Log file    → {log_path}")
    logger.info("=" * 70)

    epochs_x = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title, color in zip(
        axes,
        ["total", "sim", "reg"],
        ["Total Loss", "Similarity Loss (MSE)", "Smoothness Loss (L1)"],
        ["blue", "green", "red"]
    ):
        ax.plot(epochs_x, history[f"train_{key}"], color=color,       label="Train")
        ax.plot(epochs_x, history[f"val_{key}"],   color=color, ls="--", label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.suptitle(f"{MODEL_NAME.upper()} Registration Training", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "training_losses.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss plot saved → {plot_path}")
