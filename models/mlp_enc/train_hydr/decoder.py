from datetime import timedelta
import torch
from einops import rearrange
from torch import nn
from torchvision.ops import MLP

from aurora.batch import Batch, Metadata
from aurora.normalisation import normalise_atmos_var, normalise_surf_var, unnormalise_atmos_var, unnormalise_surf_var
from aurora.model.fourier import levels_expansion
from aurora.model.perceiver import PerceiverResampler
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)


__all__ = ["Perceiver3DDecoderLite"]


class Decoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars_new: tuple[str, ...],
        patch_size: int = 4,
        encoder_dim: int = 256,
        embed_dim: int = 1024,
        hidden_dims: list = [],
        stats = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars_new = surf_vars_new # list of strings
        self.encoder_dim = int(encoder_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.stats = stats

        self.encoders = nn.ParameterDict({
            name: nn.Sequential(
                nn.Linear(int(self.patch_size * self.patch_size), self.encoder_dim),
                nn.LayerNorm(self.encoder_dim),
                nn.GELU()
            ) for name in self.surf_vars_new
        })
        
        self.surf_heads = nn.ParameterDict(
            {name: MLP(int(self.embed_dim + self.encoder_dim), self.hidden_dims + [self.patch_size**2]) for name in self.surf_vars_new}
        )

        self.apply(init_weights)
        
    def patchify(
        self,
        x,
        B = 1,
        V = 1,
        C = 1,
    ):
        P = self.patch_size
        H = x.shape[-2] // P
        W = x.shape[-1] // P

        x = x.reshape(shape=(B, V, C, H, P, W, P))
        x = rearrange(x, "B V C H P1 W P2 -> B H W C P1 P2 V")
        x = x.reshape(shape=(B, H * W, C, V * P**2))
        return x

    def forward(
        self,
        input_dict: dict,
        latent_decoder: torch.Tensor,
        lat, # np.array
        lon, # np.array
        lsm_mask, # np.array(bool)
    ) -> Batch:

        device = latent_decoder.device

        H, W = lat.shape[0], lon.shape[0]

        input_dict = {k: normalise_surf_var(v, k, stats=self.stats) for k, v in input_dict.items()}

        # Decode surface vars. Run the head for every surface-level variable.
        latent_surf = latent_decoder[..., :1, :]
        res = []
        for name in self.surf_vars_new:
            input_var = self.patchify(input_dict[name])
            latent_new = self.encoders[name](input_var)
            latent_fusion = torch.cat([latent_surf, latent_new], dim=-1).contiguous()
            res.append(self.surf_heads[name](latent_fusion))
        x_surf = torch.stack(res, dim=-1)
        x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)
        surf_preds = unpatchify(x_surf, len(self.surf_vars_new), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)       

        pred_new = {v: surf_preds[:, i] for i, v in enumerate(self.surf_vars_new)}

        # Insert history dimension (I keep the syntax of the original code, there are some do/undo operations
        pred_new = {k: v[:, None] for k, v in pred_new.items()}

        pred_new = {k: unnormalise_surf_var(v, k, stats=self.stats) for k, v in pred_new.items()}

        for k in pred_new.keys():
            if k in ('swvl1'):
                pred_new[k][..., ~lsm_mask] = 0.
            
        return pred_new
