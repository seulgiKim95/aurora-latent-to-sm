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


class MLPDecoderLite(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars_new: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        hidden_dims: list = [],
        stats = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars_new = surf_vars_new # list of strings
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.stats = stats
        
        self.surf_heads = nn.ParameterDict(
            {name: MLP(self.embed_dim, self.hidden_dims + [self.patch_size**2]) for name in self.surf_vars_new}
        )

        self.apply(init_weights)
        

    def forward(
        self,
        latent_decoder: torch.Tensor,
        lat, # np.array
        lon, # np.array
    ) -> Batch:

        device = latent_decoder.device

        H, W = lat.shape[0], lon.shape[0]

        # Decode surface vars. Run the head for every surface-level variable.
        x_surf = torch.stack([self.surf_heads[name](latent_decoder[..., :1, :]) for name in self.surf_vars_new], dim=-1)
        x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)
        surf_preds = unpatchify(x_surf, len(self.surf_vars_new), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)       

        pred_new = {v: surf_preds[:, i] for i, v in enumerate(self.surf_vars_new)}

        # Insert history dimension (I keep the syntax of the original code, there are some do/undo operations
        pred_new = {k: v[:, None] for k, v in pred_new.items()}

        pred_new = {k: unnormalise_surf_var(v, k, stats=self.stats) for k, v in pred_new.items()}
        
        return pred_new
