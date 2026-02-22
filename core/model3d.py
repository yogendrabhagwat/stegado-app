"""
3D CNN Steganography Model for voxel data.
Falls back to universal model (flatten → 1D) when GPU is limited.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class Res3DBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm3d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.block(x))


    class Encoder3D(nn.Module):
        """
        Embeds secret into a 3D voxel cover.
        Input:  cover (B, C, D, H, W),  secret_vec (B, F)
        Output: stego  (B, C, D, H, W)
        """

        def __init__(self, in_channels=1, secret_dim=128):
            super().__init__()
            self.cover_enc = nn.Sequential(
                nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                Res3DBlock(32),
                nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                Res3DBlock(64),
            )
            self.secret_proj = nn.Sequential(
                nn.Linear(secret_dim, 64 * 4 * 4 * 4),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                Res3DBlock(64),
                nn.Conv3d(64, in_channels, 3, padding=1),
                nn.Tanh(),
            )

        def forward(self, cover, secret_vec):
            B = cover.shape[0]
            feat = self.cover_enc(cover)
            sec = self.secret_proj(secret_vec).view(B, 64, 4, 4, 4)
            sec = F.interpolate(sec, size=feat.shape[2:], mode='trilinear', align_corners=False)
            fused = torch.cat([feat, sec], dim=1)
            residual = self.decoder(fused)
            residual = F.interpolate(residual, size=cover.shape[2:], mode='trilinear', align_corners=False)
            return torch.clamp(cover + 0.1 * residual, -1, 1)


    class Decoder3D(nn.Module):
        """Extracts secret vector from 3D stego voxel."""

        def __init__(self, in_channels=1, secret_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                Res3DBlock(32),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                Res3DBlock(64),
                nn.AdaptiveAvgPool3d(2),
                nn.Flatten(),
                nn.Linear(64 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, secret_dim),
                nn.Sigmoid(),
            )

        def forward(self, stego):
            return self.net(stego)

else:
    class Encoder3D:
        pass

    class Decoder3D:
        pass
