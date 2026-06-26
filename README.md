# Learning to Unify Deformable Shape and Texture Representations for Cardiac Video Classification


Deformable shape representations have proven to be robust complements to texture features in cardiac image classification, offering geometric priors that are invariant to imaging artifacts and intensity variations. However, existing deep networks perform naive concatenation to combine these distinct feature representations, which neither exploits their complementary nature nor learns cross-modal feature dependencies. Furthermore, this results in uniform attention across all timepoints, ignoring the varying diagnostic importance across the cardiac phases. In this paper, we propose a novel cardiac video classification model that, for the first time, \textit{learns temporal features in an integrated space of deformable shape and image texture representations}. In particular, we design a bi-directional cross-attention in the latent space to fuse latent deformable shape and image features, allowing each modality to adaptively weight the other based on spatio-temporal correspondence. Unlike concatenation-based methods that apply uniform weighting across all the cardiac phases, our approach learns to dynamically adjust the contributions of shape and texture representations, derived from images, at each timepoint. We demonstrate state-of-the-art classification performance on a cine CMR video dataset, achieving improved interpretability from attention mechanisms that identify diagnostically critical cardiac phases and modality contributions.

## Setup
```bash
pip install torch torchvision timm SimpleITK scikit-learn numpy matplotlib tqdm
```

Set the paths (`TRAIN_DIR`, `TEST_DIR`) at the top of the training scripts.

## 1. Train the registration model (shape encoder)
```bash
python train_registration.py --model baseline   # VoxelMorph
python train_registration.py --model tlrn        # TLRN
```
Produces a checkpoint in `./saved_models_<model>/`.

## 2. Train the classifier
Set `REG_CKPT` (path to the checkpoint from step 1), `MODE`
(`shape` | `image` | `fusion`), and `ENCODER`
(`resnet` | `efficientnet` | `densenet` | `vit` | `vivit`) in
`train_classification.py`, then:
```bash
python train_classification.py
```
## Testing

Evaluate a trained classifier checkpoint on the test set:

```bash
python test_classifier.py --ckpt ./saved_clf_fusion_densenet/best_fusion_enc-densenet_query-gated.pth \
                          --mode fusion --encoder densenet --reg_model baseline \
                          --reg_ckpt .../baseline_best.pth --test_dir .../test
```

Arguments:
- `--ckpt`      : trained classifier checkpoint (`.pth`)
- `--mode`      : `shape` | `image` | `fusion`
- `--encoder`   : `resnet` | `efficientnet` | `densenet` | `vit` | `vivit`
- `--reg_model` : registration backbone — `baseline` | `tlrn`
- `--reg_ckpt`  : registration checkpoint matching `--reg_model`
- `--test_dir`  : path to test data (`test/{0,1}/*.mhd`)

Reports Accuracy, Precision, Recall, F1, AUC, Sensitivity, and Specificity.

## Notes
- **Registration backbones:** the `baseline` model is a VoxelMorph-style network. You can use the included `baseline_model.py`, or train the same model with the official VoxelMorph implementation: https://github.com/voxelmorph/voxelmorph
- **TLRN:** for the `tlrn` registration backbone, refer to https://github.com/nellie689/TLRN and place `tlrn_model.py` in the repo root.
- **Pretrained weights:** ImageNet weights for the image encoders download automatically on first run (cached to `./pretrained_weights/`).
- **Config matching:** when testing, `--mode`, `--encoder`, `--reg_model`, and `--reg_ckpt` must match the configuration used to train the checkpoint.
- **Data layout:** organize `.mhd` sequences (≥24 frames) as `train/{0,1}/*.mhd` and `test/{0,1}/*.mhd`.
