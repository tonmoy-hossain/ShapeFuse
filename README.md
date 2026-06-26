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

## Notes
- TLRN model code: https://github.com/nellie689/TLRN
  (place `tlrn_model.py` in the repo root)
- ImageNet weights for the image encoders download automatically on first run.
