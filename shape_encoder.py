import torch
import torch.nn as nn
from baseline_model import BaselineRegistrationModel
from tlrn_model import TLRNModel


class ShapeEncoder(nn.Module):
    def __init__(self, reg_ckpt, reg_model_kwargs, freeze=True, model_type='baseline'):
        super().__init__()
        assert model_type in ('baseline', 'tlrn')
        self.model_type = model_type

        if model_type == 'baseline':
            self.reg_model = BaselineRegistrationModel(**reg_model_kwargs)
        else:
            self.reg_model = TLRNModel(**reg_model_kwargs)

        if reg_ckpt:
            self.reg_model.load_state_dict(torch.load(reg_ckpt, map_location='cpu'))
            print(f"  [ShapeEncoder] Loaded {model_type} ckpt: {reg_ckpt}")

        if freeze:
            for p in self.reg_model.parameters():
                p.requires_grad = False
            self.reg_model.eval()

        with torch.no_grad():
            inshape = reg_model_kwargs['inshape']
            dummy   = torch.zeros(1, 2, 1, *inshape)
            if model_type == 'baseline':
                d = torch.zeros(1, 1, *inshape)
                _, _, latent = self.reg_model(d, d)
            else:
                _, _, latent_list = self.reg_model(dummy)
                latent = latent_list[0]

        self.feat_shape = latent.shape[1:]
        self.feat_dim   = latent.numel() // latent.shape[0]

    def forward(self, video):
        B, T, C, H, W = video.shape
        source = video[:, 0]
        latents = []

        if self.model_type == 'baseline':
            for t in range(1, T):
                _, _, z = self.reg_model(source, video[:, t])
                latents.append(z)
        else:
            _, _, latent_list = self.reg_model(video)
            latents = latent_list

        return torch.stack(latents, dim=1)
