import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers import ResizeTransform, VecInt, SpatialTransformer
from modelio import LoadableModel, store_config_args


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ]
    return nb_features


class ConvBlock(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        Conv = getattr(nn, f'Conv{ndims}d')
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))


class Unet(nn.Module):
    def __init__(self,
                 inshape,
                 infeats,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        self.half_res = half_res

        if nb_features is None:
            nb_features = default_unet_features()

        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide nb_levels if nb_features is int')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling    = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf

    def forward(self, x):
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        self.bottleneck = x

        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        for conv in self.remaining:
            x = conv(x)

        return x


class BaselineRegistrationModel(LoadableModel):
    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):

        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias   = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate  = VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        latent = self.unet_model.bottleneck

        flow_field = self.flow(x)

        if self.resize:
            flow_field = self.resize(flow_field)

        pos_flow = self.integrate(flow_field) if self.integrate else flow_field

        if self.fullsize:
            pos_flow = self.fullsize(pos_flow)

        warped = self.transformer(source, pos_flow)
        return warped, pos_flow, latent
