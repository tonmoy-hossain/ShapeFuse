import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_encoder import ShapeEncoder
from image_encoder import build_image_encoder


class TemporalAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, return_weights=False):
        w = torch.softmax(self.score(x), dim=1)
        pooled = (x * w).sum(dim=1)
        if return_weights:
            return pooled, w.squeeze(-1)
        return pooled


class BahdanauPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_h = nn.Linear(dim, dim)
        self.w   = nn.Linear(dim, 1, bias=False)

    def forward(self, x, return_weights=False):
        e = self.w(torch.tanh(self.W_h(x)))
        alpha = torch.softmax(e, dim=1)
        h = (x * alpha).sum(dim=1)
        if return_weights:
            return h, alpha.squeeze(-1)
        return h


def _mlp_head(in_dim, hidden_dim, num_classes, dropout=0.3):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class ShapeClassifier(nn.Module):
    def __init__(self, shape_encoder_kwargs, num_classes=2, hidden_dim=256):
        super().__init__()
        self.encoder = ShapeEncoder(**shape_encoder_kwargs)

        C, lH, lW = self.encoder.feat_shape
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = C * 4 * 4

        self.proj       = nn.Linear(flat_dim, hidden_dim)
        self.temp_pool  = TemporalAttnPool(hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_classes)

    def forward(self, video):
        latents = self.encoder(video)
        B, Tm1, C, lH, lW = latents.shape

        z = latents.view(B * Tm1, C, lH, lW)
        z = self.spatial_pool(z).flatten(1)
        z = F.relu(self.proj(z))
        z = z.view(B, Tm1, -1)

        z = self.temp_pool(z)
        return self.classifier(z)


class ImageClassifier(nn.Module):
    def __init__(self, encoder_name, encoder_kwargs=None, num_classes=2, hidden_dim=256):
        super().__init__()
        self.encoder = build_image_encoder(encoder_name, **(encoder_kwargs or {}))

        feat_dim = self.encoder.feat_dim

        if not self.encoder.is_video_enc:
            self.temp_pool = TemporalAttnPool(feat_dim)
        else:
            self.temp_pool = None

        self.classifier = _mlp_head(feat_dim, hidden_dim, num_classes)

    def forward(self, video):
        feats = self.encoder(video)
        if self.temp_pool is not None:
            feats = self.temp_pool(feats)
        return self.classifier(feats)


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_seq, kv_seq, return_attn=False):
        out, attn_weights = self.attn(query_seq, kv_seq, kv_seq,
                                      need_weights=True,
                                      average_attn_weights=True)
        out = self.norm(out + query_seq)
        if return_attn:
            return out, attn_weights
        return out


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, a, b, return_gate=False):
        g = self.gate(torch.cat([a, b], dim=-1))
        fused = g * a + (1 - g) * b
        if return_gate:
            return fused, g
        return fused


class FusionClassifier(nn.Module):
    def __init__(self,
                 shape_encoder_kwargs,
                 encoder_name,
                 encoder_kwargs=None,
                 num_classes=2,
                 hidden_dim=256,
                 num_heads=8):
        super().__init__()

        self.shape_enc  = ShapeEncoder(**shape_encoder_kwargs)
        C, lH, lW      = self.shape_enc.feat_shape
        self.shape_pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim        = C * 4 * 4
        self.shape_proj = nn.Linear(flat_dim, hidden_dim)

        self.image_enc  = build_image_encoder(encoder_name, **(encoder_kwargs or {}))
        img_dim         = self.image_enc.feat_dim
        self.image_proj = nn.Linear(img_dim, hidden_dim)

        if self.image_enc.is_video_enc:
            print("  [FusionClassifier] Warning: video encoders collapse time and cannot be "
                  "per-timepoint gated. Use resnet / efficientnet / densenet / vit.")

        self.shape_queries_image = CrossAttnBlock(hidden_dim, num_heads)
        self.image_queries_shape = CrossAttnBlock(hidden_dim, num_heads)

        self.gated_fusion = GatedFusion(hidden_dim)
        self.pool         = BahdanauPool(hidden_dim)

        self.classifier = _mlp_head(hidden_dim, hidden_dim, num_classes)

    def _get_shape_seq(self, video):
        latents = self.shape_enc(video)
        B, Tm1, C, lH, lW = latents.shape
        z = latents.view(B * Tm1, C, lH, lW)
        z = self.shape_pool(z).flatten(1)
        z = F.relu(self.shape_proj(z))
        z = z.view(B, Tm1, -1)
        pad = z.new_zeros(B, 1, z.shape[-1])
        return torch.cat([pad, z], dim=1)

    def _get_image_seq(self, video):
        feats = self.image_enc(video)
        if self.image_enc.is_video_enc:
            feats = feats.unsqueeze(1)
        return F.relu(self.image_proj(feats))

    def forward(self, video, return_attn=False):
        shape_seq = self._get_shape_seq(video)
        image_seq = self._get_image_seq(video)

        if return_attn:
            z_v, s2i_attn = self.shape_queries_image(shape_seq, image_seq, return_attn=True)
            z_f, i2s_attn = self.image_queries_shape(image_seq, shape_seq, return_attn=True)
            fused_seq, gate = self.gated_fusion(z_v, z_f, return_gate=True)
            h, alpha = self.pool(fused_seq, return_weights=True)
            logits = self.classifier(h)
            attn_dict = {
                's2i_attn': s2i_attn,
                'i2s_attn': i2s_attn,
                'gate'    : gate,
                'alpha'   : alpha,
            }
            return logits, attn_dict

        z_v = self.shape_queries_image(shape_seq, image_seq)
        z_f = self.image_queries_shape(image_seq, shape_seq)
        fused_seq = self.gated_fusion(z_v, z_f)
        h = self.pool(fused_seq)
        return self.classifier(h)
