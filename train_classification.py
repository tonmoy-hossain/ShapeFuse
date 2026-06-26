import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import random_split
import torch.nn.functional as F
from tqdm import tqdm
import SimpleITK as sitk

from classifier import ShapeClassifier, ImageClassifier, FusionClassifier

TRAIN_DIR   = ".../train"
TEST_DIR    = ".../test"
IMG_SIZE    = (128, 128)
CROP_DIST   = 25
BATCH_SIZE  = 8
EPOCHS      = 35
LR          = 1e-4
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE        = "fusion"
ENCODER     = "densenet"
REG_CKPT    = ".../tlrn_best.pth"

ENCODER_KWARGS = {'model_name': 'efficientnet_b0'} if ENCODER == 'efficientnet' else \
                 {'num_frames': 24}                 if ENCODER == 'vivit'        else \
                 {}

SAVE_DIR    = f"./saved_clf_{MODE}_{ENCODER}"
os.makedirs(SAVE_DIR, exist_ok=True)


REG_MODEL_KWARGS = dict(
    inshape                = IMG_SIZE,
    nb_unet_features       = [[16, 32], [32, 32, 16, 16]],
    nb_unet_conv_per_level = 1,
    int_steps              = 7,
    int_downsize           = 2,
    src_feats              = 1,
    trg_feats              = 1,
    unet_half_res          = True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SAVE_DIR, "train.log")),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class CardiacVideoDataset(data.Dataset):
    def __init__(self, root_dir, img_size=IMG_SIZE, crop_dist=CROP_DIST):
        self.img_size  = img_size
        self.crop_dist = crop_dist
        self.samples   = []

        for label_str in ['0', '1']:
            folder = os.path.join(root_dir, label_str)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.mhd'):
                    fpath = os.path.join(folder, fname)
                    arr = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
                    if arr.shape[0] >= 24:
                        self.samples.append((fpath, int(label_str)))

        print(f"  Dataset: {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(fpath)).astype(np.float32)
        arr = arr[:24, self.crop_dist:-self.crop_dist, self.crop_dist:-self.crop_dist]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        vol = torch.tensor(arr).unsqueeze(1)
        vol = F.interpolate(vol, size=self.img_size, mode='bilinear', align_corners=False)
        return vol, label


train_val_ds = CardiacVideoDataset(TRAIN_DIR)
n_val        = int(len(train_val_ds) * 0.2)
n_train      = len(train_val_ds) - n_val
train_ds, val_ds = random_split(train_val_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = data.DataLoader(CardiacVideoDataset(TEST_DIR),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"  Train: {n_train} | Val: {n_val} | Test: {len(test_loader.dataset)}")

shape_enc_kwargs = dict(
    reg_ckpt         = REG_CKPT,
    reg_model_kwargs = REG_MODEL_KWARGS,
    freeze           = True,
    model_type       = 'tlrn',
)

if MODE == "shape":
    model = ShapeClassifier(shape_encoder_kwargs=shape_enc_kwargs,
                            num_classes=NUM_CLASSES, hidden_dim=256)

elif MODE == "image":
    model = ImageClassifier(encoder_name=ENCODER, encoder_kwargs=ENCODER_KWARGS,
                            num_classes=NUM_CLASSES, hidden_dim=256)

elif MODE == "fusion":
    model = FusionClassifier(shape_encoder_kwargs=shape_enc_kwargs,
                             encoder_name=ENCODER, encoder_kwargs=ENCODER_KWARGS,
                             num_classes=NUM_CLASSES, hidden_dim=256)

model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Mode: {MODE} | Encoder: {ENCODER} | Trainable params: {n_params:,}\n")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0., 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for videos, labels in loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            logits = model(videos)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


def run_test(loader):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            logits = model(videos)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc


_enc     = ENCODER if MODE != "shape"  else "none"
_qm      = "gated" if MODE == "fusion" else "none"
RUN_NAME = f"{MODE}_enc-{_enc}_query-{_qm}"

best_val_acc = 0.
logger.info(f"{'Ep':>5} | {'TrLoss':>7} {'TrAcc':>6} | {'VlLoss':>7} {'VlAcc':>6} | Best")

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    vl_acc, vl_prec, vl_rec, vl_f1, vl_auc = run_test(val_loader)
    scheduler.step()

    is_best = vl_acc > best_val_acc
    if is_best:
        best_val_acc = vl_acc
        torch.save({
            "model_state_dict" : model.state_dict(),
            "mode"             : MODE,
            "encoder"          : _enc,
            "query_mode"       : _qm,
            "epoch"            : epoch + 1,
            "best_val_acc"     : best_val_acc,
        }, os.path.join(SAVE_DIR, f"best_{RUN_NAME}.pth"))

    logger.info(f"{epoch+1:>5} | {tr_loss:>7.4f} {tr_acc:>6.3f} | "
                f"Acc: {vl_acc:.3f} Prec: {vl_prec:.3f} "
                f"Rec: {vl_rec:.3f} F1: {vl_f1:.3f} AUC: {vl_auc:.3f} "
                f"| {'*' if is_best else ''}")

    if (epoch + 1) % 5 == 0:
        acc, prec, rec, f1, auc = run_test(test_loader)
        logger.info(f"  [Test @ Ep {epoch+1:>3}] "
                    f"Acc: {acc:.3f} | Prec: {prec:.3f} | "
                    f"Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

torch.save({
    "model_state_dict" : model.state_dict(),
    "mode"             : MODE,
    "encoder"          : _enc,
    "query_mode"       : _qm,
    "epoch"            : EPOCHS,
    "best_val_acc"     : best_val_acc,
}, os.path.join(SAVE_DIR, f"final_{RUN_NAME}.pth"))
logger.info(f"Training done. Best val acc: {best_val_acc:.3f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ckpt = torch.load(os.path.join(SAVE_DIR, f"best_{RUN_NAME}.pth"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for videos, labels in test_loader:
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        logits = model(videos)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec  = recall_score(all_labels, all_preds, zero_division=0)
f1   = f1_score(all_labels, all_preds, zero_division=0)
auc  = roc_auc_score(all_labels, all_probs)

logger.info(f"Test  Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
print(f"\n{'Metric':<12} {'Score':>6}")
print("-" * 20)
for name, val in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1-Score", f1), ("AUC", auc)]:
    print(f"{name:<12} {val:>6.3f}")
