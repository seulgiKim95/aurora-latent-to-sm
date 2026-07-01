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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(
        self,
        surf_vars_new: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 512,
        sm_key: str = 'swvl1',
    ) -> None:
        super().__init__()
        self.surf_vars_new = surf_vars_new # for now, only sm is supported
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sm_key = sm_key
        self.len_vars = len(surf_vars_new)

        # 1. Preprocessing for the high-dimensional latent (preserving information)
        self.latent_pre_proc = nn.Sequential( # 2E -> E -> E -> E/2
            nn.Conv2d(2*self.embed_dim, self.embed_dim, kernel_size=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1), # relearn spatial relationships
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim//2, kernel_size=4, stride=2, padding=1)
        )

        # 2. SM feature extraction (expand the channel count to match the latent)
        # (low-resolution features for the skip connection)
        self.sm_low_enc = nn.Sequential(
            nn.Conv2d(self.len_vars, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # (high-dimensional features for fusion - downsampling)
        self.sm_high_enc = nn.Sequential(
            nn.Conv2d(self.len_vars, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.embed_dim//2, kernel_size=3, padding=1, stride=2)
        )

        # 3. Fusion Block (256 + 256 = 512 channels)
        self.fusion_layer = nn.Sequential( # E -> E/2 -> E/2 -> E/4
            nn.Conv2d(self.embed_dim, self.embed_dim//2, kernel_size=3, padding=1),
            ResidualBlock(self.embed_dim//2, self.embed_dim//2), # use the ResidualBlock defined above
            nn.ConvTranspose2d(self.embed_dim//2, self.embed_dim//4, kernel_size=2, stride=2)
        )

        # 4. Final Delta Head
        self.final_head = nn.Conv2d(self.embed_dim//4 + 64, self.len_vars, kernel_size=3, padding=1)
        pass

    def forward(self, X, Z, lsm_mask):
        # Z: [B, L, C, 2E] -> [B, 2E * C, L] -> [B, 2E * C, H/P, W/P]
        Zp = torch.einsum('BLCE -> BECL', Z[..., :1, :])
        Z = Zp.reshape(Z.shape[0], 2 * self.embed_dim * 1, X[self.sm_key].shape[-2] // self.patch_size, X[self.sm_key].shape[-1] // self.patch_size)
        # Z: assume it has been reshaped to [B, 1024, H/4, W/4]

        # X: normalize
        norm_X = {k: normalise_surf_var(v, k) for k, v in X.items()}

        # process Aurora features
        z_feat = self.latent_pre_proc(Z) # [B, 256, H/2, W/2]

        # process SM features
        sm_low = self.sm_low_enc(norm_X[self.sm_key]) # [B, 64, H, W]
        sm_high = self.sm_high_enc(norm_X[self.sm_key]) # [B, 256, H/2, W/2]

        # feature fusion
        combined = torch.cat([z_feat, sm_high], dim=1) # [B, 512, H/2, W/2]
        delta_feat = self.fusion_layer(combined) # [B, 128, H, W]

        # combine the skip connection and produce the final output
        # re-check delta_feat against the size of X in case the resolutions differ
        # if delta_feat.shape[-2:] != sm_low.shape[-2:]:
        #     delta_feat = F.interpolate(delta_feat, size=sm_low.shape[-2:], mode='bilinear', align_corners=False)

        merged = torch.cat([delta_feat, sm_low], dim=1) # [B, 192, H, W]
        final_delta = self.final_head(merged) # [B, V, H, W]

        pred_new = {}
        for k in self.surf_vars_new:
            pred_new[k] = final_delta

        pred_new = {k: unnormalise_surf_var(v, k) for k, v in pred_new.items()}

        for k in pred_new.keys():
            if k in ('swvl1'):
                pred_new[k][..., ~lsm_mask] = 0.

        return pred_new
