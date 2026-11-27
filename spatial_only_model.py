# spatial_only_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import ResBlock3D


class Model_SpatialOnlySR(nn.Module):
    """
    Reines Spatial-Super-Resolution 3D-U-Net (Ablation zu ModelA_FlexibleSR).
    Identische Backbone-Tiefe, ResBlocks und Kanalbreiten (16 -> 32 -> 64),
    jedoch ohne FiLM-Layer, ohne Metadaten und ohne masked_average.
    
    Input:  [B, in_channels, D_lr, H_lr, W_lr]  (Standard: Kanal 0 = LR-DWI, Kanal 1 = LR-b0)
    Output: [B, 1, D_hr, H_hr, W_hr]            (HR-Rekonstruktion derselben DWI-Richtung)
    """
    def __init__(self, in_channels=2):
        super().__init__()
        self.in_channels = in_channels

        # --- Encoder (identisch zu ModelA_FlexibleSR) ---
        self.init_conv = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(ResBlock3D(16), ResBlock3D(16))

        self.down1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))

        self.down2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Sequential(ResBlock3D(64), ResBlock3D(64))

        # --- Decoder (klassische U-Net Skip-Connections) ---
        # 64 (Bottleneck) + 32 (Skip e2) = 96 -> 32
        self.dec2_conv = nn.Conv3d(96, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))

        # 32 (von unten) + 16 (Skip e1) = 48 -> 16
        self.dec1_conv = nn.Conv3d(48, 16, kernel_size=3, padding=1)
        self.dec1 = nn.Sequential(ResBlock3D(16), ResBlock3D(16))

        # --- Upsampling Head (LR -> HR, Faktor 2) ---
        self.head_conv = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
        )

    def upsample_trilinear(self, x):
        return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(self.init_conv(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Decoder
        d2 = self.upsample_trilinear(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(self.dec2_conv(d2))

        d1 = self.upsample_trilinear(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(self.dec1_conv(d1))

        # HR Head
        out = self.upsample_trilinear(d1)
        out = self.head_conv(out)
        return out