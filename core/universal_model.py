"""
Universal Multi-Modal Steganography Model
Architecture: ResNet Encoder + UNet Decoder with Spatial & Channel Attention.
Handles: Text, 2D Image, and 3D flatten → tensor secrets.
Requires PyTorch (optional – system falls back to signal-processing engine).
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ChannelAttention(nn.Module):
        def __init__(self, channels, ratio=8):
            super().__init__()
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.max = nn.AdaptiveMaxPool2d(1)
            mid = max(1, channels // ratio)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(mid, channels, 1, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return x * self.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x)))


    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg = x.mean(dim=1, keepdim=True)
            mx, _ = x.max(dim=1, keepdim=True)
            atten = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
            return x * atten


    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.ca = ChannelAttention(channels)
            self.sa = SpatialAttention()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            r = self.sa(self.ca(self.block(x)))
            return self.relu(x + r)


    class SecretBranch(nn.Module):
        """Encodes variable-length secret tensor into a spatial feature map."""
        def __init__(self, secret_dim=256, out_channels=64):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(secret_dim, 512),
                nn.ReLU(),
                nn.Linear(512, out_channels * 4 * 4),
                nn.ReLU(),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
                nn.ReLU(),
            )

        def forward(self, s):
            b = s.shape[0]
            x = self.proj(s).view(b, -1, 4, 4)
            return self.up(x)


    class UNetDecoder(nn.Module):
        def __init__(self, in_ch=128):
            super().__init__()
            self.up1 = self._block(in_ch, 64)
            self.up2 = self._block(64, 32)
            self.up3 = self._block(32, 16)
            self.out_conv = nn.Conv2d(16, 3, 1)
            self.tanh = nn.Tanh()

        def _block(self, inc, outc):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(inc, outc, 3, padding=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
                ResBlock(outc),
            )

        def forward(self, x):
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            return self.tanh(self.out_conv(x))


    class UniversalEncoder(nn.Module):
        """ResNet-based encoder fusing cover image + secret branch."""
        def __init__(self, secret_dim=256):
            super().__init__()
            # Cover stream
            self.cover_stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.cover_res = nn.Sequential(*[ResBlock(64) for _ in range(4)])

            # Secret branch
            self.secret_branch = SecretBranch(secret_dim, 64)

            # Fusion
            self.fusion = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.ReLU(inplace=True),
                ResBlock(128),
            )

            # Decoder
            self.decoder = UNetDecoder(128)

        def forward(self, cover, secret_vec):
            """
            cover:      (B, 3, H, W) normalized [-1,1]
            secret_vec: (B, secret_dim)
            returns:    (B, 3, H, W) stego image residual
            """
            c = self.cover_res(self.cover_stem(cover))   # (B, 64, H/2, W/2)
            s = self.secret_branch(secret_vec)            # (B, 64, 16, 16)
            # Align spatial sizes
            s = F.interpolate(s, size=c.shape[2:], mode='bilinear', align_corners=False)
            fused = self.fusion(torch.cat([c, s], dim=1))
            residual = self.decoder(fused)
            residual = F.interpolate(residual, size=cover.shape[2:], mode='bilinear', align_corners=False)
            stego = torch.clamp(cover + 0.1 * residual, -1, 1)
            return stego


    class SecretDecoder(nn.Module):
        """Extracts secret vector from stego image."""
        def __init__(self, secret_dim=256):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                ResBlock(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                ResBlock(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, secret_dim),
                nn.Sigmoid(),
            )

        def forward(self, stego):
            return self.head(self.backbone(stego))

else:
    # Stub classes when PyTorch is unavailable
    class UniversalEncoder:
        pass

    class SecretDecoder:
        pass
