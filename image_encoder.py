import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tvm
    from torchvision.models import ResNet50_Weights, DenseNet121_Weights
except ImportError:
    raise ImportError("pip install torchvision")

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")


_HERE       = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_HERE, "pretrained_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def _load_or_download_torchvision(model_fn, weights_enum, local_name):
    local_path = os.path.join(WEIGHTS_DIR, local_name)
    if os.path.exists(local_path):
        print(f"  [weights] Loading {local_name} from local cache")
        model = model_fn(weights=None)
        model.load_state_dict(torch.load(local_path, map_location='cpu'))
    else:
        print(f"  [weights] {local_name} not found locally — downloading...")
        model = model_fn(weights=weights_enum)
        torch.save(model.state_dict(), local_path)
        print(f"  [weights] Saved to {local_path}")
    return model


def _load_or_download_timm(model_name, local_name, **timm_kwargs):
    local_path = os.path.join(WEIGHTS_DIR, local_name)
    if os.path.exists(local_path):
        print(f"  [weights] Loading {local_name} from local cache")
        model = timm.create_model(model_name, pretrained=False, **timm_kwargs)
        state = torch.load(local_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    else:
        print(f"  [weights] {local_name} not found locally — downloading...")
        model = timm.create_model(model_name, pretrained=True, **timm_kwargs)
        torch.save(model.state_dict(), local_path)
        print(f"  [weights] Saved to {local_path}")
    return model


class ResNetEncoder(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        base = _load_or_download_torchvision(
            tvm.resnet50, ResNet50_Weights.IMAGENET1K_V1, "resnet50.pth"
        )
        self.first_conv = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.first_conv.weight.data = base.conv1.weight.data.mean(dim=1, keepdim=True)
        self.body = nn.Sequential(
            base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.feat_dim     = 2048
        self.is_video_enc = False
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    def encode_frame(self, x):
        return self.body(self.first_conv(x)).flatten(1)

    def forward(self, video):
        B, T, C, H, W = video.shape
        frames = video.view(B * T, C, H, W)
        feats  = self.encode_frame(frames)
        return feats.view(B, T, self.feat_dim)


class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0', freeze=False):
        super().__init__()
        local_name = f"{model_name}.pth"
        self.model = _load_or_download_timm(
            model_name, local_name, num_classes=0, in_chans=1
        )
        self.feat_dim     = self.model.num_features
        self.is_video_enc = False
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, video):
        B, T, C, H, W = video.shape
        frames = video.view(B * T, C, H, W)
        feats  = self.model(frames)
        return feats.view(B, T, self.feat_dim)


class DenseNetEncoder(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        base = _load_or_download_torchvision(
            tvm.densenet121, DenseNet121_Weights.IMAGENET1K_V1, "densenet121.pth"
        )
        old_conv = base.features.conv0
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features.conv0  = new_conv

        self.features     = base.features
        self.feat_dim     = 1024
        self.is_video_enc = False

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, video):
        B, T, C, H, W = video.shape
        frames = video.view(B * T, C, H, W)
        z = self.features(frames)
        z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        return z.view(B, T, self.feat_dim)


class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', freeze=False):
        super().__init__()
        local_name = f"{model_name}.pth"
        self.model = _load_or_download_timm(
            model_name, local_name, num_classes=0, in_chans=1
        )
        self.feat_dim     = self.model.num_features
        self.is_video_enc = False
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, video):
        B, T, C, H, W = video.shape
        frames = video.view(B * T, C, H, W)
        if H != 224:
            frames = F.interpolate(frames, (224, 224), mode='bilinear', align_corners=False)
        feats = self.model(frames)
        return feats.view(B, T, self.feat_dim)


class ViViTEncoder(nn.Module):
    def __init__(self,
                 spatial_model='vit_small_patch16_224',
                 num_frames=24,
                 temporal_depth=4,
                 num_heads=8,
                 freeze_spatial=False):
        super().__init__()
        local_name = f"{spatial_model}.pth"
        self.spatial = _load_or_download_timm(
            spatial_model, local_name, num_classes=0, in_chans=1
        )
        d = self.spatial.num_features

        if freeze_spatial:
            for p in self.spatial.parameters():
                p.requires_grad = False

        self.temp_pos = nn.Parameter(torch.randn(1, num_frames, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True,
        )
        self.temporal_tf  = nn.TransformerEncoder(enc_layer, num_layers=temporal_depth)
        self.feat_dim     = d
        self.is_video_enc = True

    def forward(self, video):
        B, T, C, H, W = video.shape
        frames = video.view(B * T, C, H, W)
        if H != 224:
            frames = F.interpolate(frames, (224, 224), mode='bilinear', align_corners=False)
        spatial_feats = self.spatial(frames).view(B, T, -1)
        x = spatial_feats + self.temp_pos[:, :T]
        x = self.temporal_tf(x)
        return x.mean(dim=1)


def build_image_encoder(name, **kwargs):
    encoders = {
        'resnet'      : ResNetEncoder,
        'efficientnet': EfficientNetEncoder,
        'densenet'    : DenseNetEncoder,
        'vit'         : ViTEncoder,
        'vivit'       : ViViTEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown encoder '{name}'. Choose from {list(encoders)}")
    return encoders[name](**kwargs)
