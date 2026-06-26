import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import SimpleITK as sitk
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from classifier import ShapeClassifier, ImageClassifier, FusionClassifier

IMG_SIZE  = (128, 128)
CROP_DIST = 25
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class CardiacVideoDataset(data.Dataset):
    def __init__(self, root_dir):
        self.samples = []
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
        labels = [s[1] for s in self.samples]
        print(f"  Test samples: {len(self.samples)}  "
              f"(class 0: {labels.count(0)}  class 1: {labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(fpath)).astype(np.float32)
        arr = arr[:24, CROP_DIST:-CROP_DIST, CROP_DIST:-CROP_DIST]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        vol = torch.tensor(arr).unsqueeze(1)
        vol = F.interpolate(vol, size=IMG_SIZE, mode='bilinear', align_corners=False)
        return vol, label


def build_model(args):
    shape_enc_kwargs = dict(
        reg_ckpt         = args.reg_ckpt,
        reg_model_kwargs = REG_MODEL_KWARGS,
        freeze           = True,
        model_type       = args.reg_model,
    )
    encoder_kwargs = {'model_name': 'efficientnet_b0'} if args.encoder == 'efficientnet' else \
                     {'num_frames': 24}                 if args.encoder == 'vivit'        else \
                     {}

    if args.mode == "shape":
        return ShapeClassifier(shape_encoder_kwargs=shape_enc_kwargs,
                               num_classes=2, hidden_dim=args.hidden_dim)
    if args.mode == "image":
        return ImageClassifier(encoder_name=args.encoder, encoder_kwargs=encoder_kwargs,
                               num_classes=2, hidden_dim=args.hidden_dim)
    return FusionClassifier(shape_encoder_kwargs=shape_enc_kwargs,
                            encoder_name=args.encoder, encoder_kwargs=encoder_kwargs,
                            num_classes=2, hidden_dim=args.hidden_dim)


def compute_metrics(labels, preds, probs):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "Accuracy"   : accuracy_score(labels, preds),
        "Precision"  : precision_score(labels, preds, zero_division=0),
        "Recall"     : recall_score(labels, preds, zero_division=0),
        "F1"         : f1_score(labels, preds, zero_division=0),
        "AUC"        : roc_auc_score(labels, probs),
        "Sensitivity": tp / (tp + fn + 1e-8),
        "Specificity": tn / (tn + fp + 1e-8),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--test_dir",   type=str, default=".../test")
    parser.add_argument("--mode",       type=str, default="fusion",
                        choices=["shape", "image", "fusion"])
    parser.add_argument("--encoder",    type=str, default="densenet",
                        choices=["resnet", "efficientnet", "densenet", "vit", "vivit"])
    parser.add_argument("--reg_model",  type=str, default="tlrn",
                        choices=["baseline", "tlrn"])
    parser.add_argument("--reg_ckpt",   type=str, default=".../tlrn_best.pth")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"\nLoading classifier from: {args.ckpt}")
    model = build_model(args).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    print(f"Loading test data from: {args.test_dir}")
    loader = data.DataLoader(CardiacVideoDataset(args.test_dir),
                             batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(DEVICE)
            logits = model(videos)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    print(f"\n{'-' * 40}")
    print(f"  Ckpt    : {os.path.basename(args.ckpt)}")
    print(f"  Mode    : {args.mode}  |  Encoder: {args.encoder}  |  Reg: {args.reg_model}")
    print(f"{'-' * 40}")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")
    print(f"{'-' * 40}\n")
