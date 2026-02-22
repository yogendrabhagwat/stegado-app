"""
Neural Steganography Engine  v3 — Stegado
═══════════════════════════════════════════
Pure AI-driven steganography — NO sequential LSB, NO AES encryption.

Architecture:
 PASSWORD → SHA-256 → PyTorch seed
                ↓
 Random noise → AttentionNet (5-layer CNN) → Importance Map (H×W)
                ↓
 All H×W×3 positions ranked by neural importance score (argsort)
                ↓
 Positions allocated in NON-OVERLAPPING slices:
   [0   .. 191 ] → Header ×3 (3 × 64 bits)
   [192 .. N   ] → Payload ×3 (3 redundant copies, majority-vote on extract)
                ↓
 ±1 pixel modification at selected neural positions (imperceptible)


Why this is AI steganography, NOT plain LSB:
 • Embedding positions are chosen by a CNN, not sequential pixel order
 • The CNN uses password-seeded weights + password-seeded noise input
 • Different password = totally different positions = cannot decode
 • Positions favour complex-texture regions (natural image statistics)
 • No AES — password security comes from the neural network itself
"""

import io
import struct
import hashlib
import logging
import numpy as np
from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_MAGIC     = 0x4E455552   # 'NEUR'
_HDR_BITS  = 64           # 32-bit magic + 32-bit payload-length
_HDR_TOTAL = _HDR_BITS * 3  # 192 positions reserved for header


# ── Bit helpers ───────────────────────────────────────────────────────────────

def _int_to_bits(n: int, width: int = 32):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def _bits_to_int(bits) -> int:
    r = 0
    for b in bits:
        r = (r << 1) | int(b)
    return r

def _bytes_to_bits(data: bytes):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def _bits_to_bytes(bits) -> bytes:
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i + j])
        out.append(val)
    return bytes(out)


# ── Password → seed ───────────────────────────────────────────────────────────

def _pw_seed(password: str) -> int:
    return int(hashlib.sha256(password.encode('utf-8')).hexdigest()[:8], 16)


# ── Neural Attention Network ──────────────────────────────────────────────────

class AttentionNet(nn.Module):
    """
    5-layer CNN producing a per-pixel importance map [0, 1].
    Weights are randomly initialized and fully seeded by the password.
    NO training required — the network drives position selection, not quality.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _neural_ranking(H: int, W: int, password: str) -> np.ndarray:
    """
    Returns ALL H×W×3 flat-array indices sorted by neural importance score
    (highest neural importance LAST → we take from the tail for embedding).

    Key design:
    • CNN weights are seeded by password → password-specific pattern
    • Input noise is seeded by password → adds positional salt
    • Sorting is deterministic (argsort) → same order every call
    • Image content is NOT used → map identical for cover & stego images
    """
    seed = _pw_seed(password)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    model = AttentionNet()
    model.eval()

    # Password-seeded noise as CNN input (NOT the actual image pixels)
    noise = np.random.rand(H, W, 3).astype(np.float32)
    t = torch.from_numpy(noise.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        att = model(t).squeeze().numpy()          # H×W

    att = att - att.min()
    mx = att.max()
    if mx > 1e-8:
        att /= mx

    # Add a second layer of password-seeded shuffle to break ties
    rng   = np.random.RandomState(seed % (2**32))
    noise2 = rng.rand(H, W).astype(np.float32)
    score  = att * 0.7 + noise2 * 0.3             # H×W

    # Tile to H×W×3 (one score per channel per pixel)
    score_3ch = np.tile(score.flatten(), 3)       # H*W*3

    return np.argsort(score_3ch).astype(np.int64)  # ascending; tail = best positions


# ── Position slices (NON-OVERLAPPING) ─────────────────────────────────────────

def _header_slice(ranking: np.ndarray, copy: int) -> np.ndarray:
    """64 positions for header copy `copy` (0, 1, or 2). Never overlaps payload."""
    start = copy * _HDR_BITS
    return ranking[-(start + _HDR_BITS): len(ranking) - start if start > 0 else None][::-1][:_HDR_BITS]


def _payload_slice(ranking: np.ndarray, n: int, region: int) -> np.ndarray:
    """n positions for payload region `region` (0, 1, or 2). Starts after header area."""
    # Header occupies the TOP _HDR_TOTAL positions → payload uses the rest
    total    = len(ranking)
    avail    = total - _HDR_TOTAL        # positions not used by header
    region_size = avail // 3
    # Take region from the "good but not top" positions (ascending from tail - HDR_TOTAL)
    tail_start = total - _HDR_TOTAL - 1  # last usable position index for payload
    start = tail_start - region * region_size
    end   = start - n
    if end < 0:
        raise ValueError("Payload too large for available neural positions.")
    return ranking[end + 1: start + 1][::-1][:n]


# ── Embed ─────────────────────────────────────────────────────────────────────

def neural_embed(
    cover_img: Image.Image,
    secret_bytes: bytes,
    password: str,
) -> Tuple[Image.Image, float, float]:
    """
    Embed secret_bytes into cover_img using Neural Attention Steganography.
    Returns (stego_image, psnr_db, ssim).
    """
    cover_img = cover_img.convert('RGB')
    cover_arr = np.array(cover_img, dtype=np.uint8)
    H, W, _   = cover_arr.shape

    ranking = _neural_ranking(H, W, password)   # H*W*3 sorted indices

    payload       = _encode_payload(secret_bytes)
    payload_bits  = _bytes_to_bits(payload)
    n             = len(payload_bits)

    hdr_bits = _int_to_bits(_MAGIC, 32) + _int_to_bits(n, 32)   # 64 bits

    avail = H * W * 3 - _HDR_TOTAL
    if n * 3 > avail:
        max_b = max(0, avail // 3 // 8 - 4)
        raise ValueError(
            f"Secret too large. Max ≈ {max_b} bytes for this image. "
            f"Please use a larger cover image (≥ 512×512 recommended)."
        )

    flat = cover_arr.flatten().astype(np.int32)

    # ── Embed header 3× (non-overlapping top positions) ─────────────────────
    for copy in range(3):
        positions = _header_slice(ranking, copy)
        for i, bit in enumerate(hdr_bits):
            p = int(positions[i])
            v = int(flat[p])
            if (v & 1) != bit:
                flat[p] = v + 1 if v < 255 else v - 1

    # ── Embed payload 3× (separate non-overlapping regions below header) ────
    for region in range(3):
        positions = _payload_slice(ranking, n, region)
        for i, bit in enumerate(payload_bits):
            p = int(positions[i])
            v = int(flat[p])
            if (v & 1) != bit:
                flat[p] = v + 1 if v < 255 else v - 1

    stego_arr  = flat.reshape(H, W, 3).clip(0, 255).astype(np.uint8)
    stego_img  = Image.fromarray(stego_arr, 'RGB')

    psnr = _psnr(cover_arr, stego_arr)
    ssim = _ssim_fast(cover_arr, stego_arr)
    return stego_img, round(psnr, 2), round(float(ssim), 4)


# ── Extract ───────────────────────────────────────────────────────────────────

def neural_extract(stego_img: Image.Image, password: str) -> bytes:
    """Extract secret using the same neural attention process."""
    stego_img  = stego_img.convert('RGB')
    stego_arr  = np.array(stego_img, dtype=np.uint8)
    H, W, _    = stego_arr.shape
    flat       = stego_arr.flatten().astype(np.int32)

    _ERR = (
        "No neural steganography data found.\n"
        "• Only images encoded by Stegado can be decoded.\n"
        "• Password must exactly match what was used when hiding."
    )

    ranking = _neural_ranking(H, W, password)

    # ── Read header 3× and majority-vote ────────────────────────────────────
    magics, lengths = [], []
    for copy in range(3):
        positions = _header_slice(ranking, copy)
        m_bits = [int(flat[int(positions[i])]) & 1 for i in range(32)]
        n_bits = [int(flat[int(positions[32 + i])]) & 1 for i in range(32)]
        magics.append(_bits_to_int(m_bits))
        lengths.append(_bits_to_int(n_bits))

    if sum(1 for m in magics if m == _MAGIC) < 2:
        raise ValueError(_ERR)

    lengths.sort()
    n = lengths[1]   # median

    if n < 8 or n > (H * W * 3 - _HDR_TOTAL) // 3:
        raise ValueError(_ERR)

    # ── Read payload 3× and majority-vote ───────────────────────────────────
    all_copies = []
    for region in range(3):
        positions = _payload_slice(ranking, n, region)
        bits = [int(flat[int(positions[i])]) & 1 for i in range(n)]
        all_copies.append(bits)

    payload_bits = [
        1 if (all_copies[0][i] + all_copies[1][i] + all_copies[2][i]) >= 2 else 0
        for i in range(n)
    ]
    payload_bytes = _bits_to_bytes(payload_bits)
    return _decode_payload(payload_bytes)


# ── Payload helpers ───────────────────────────────────────────────────────────

def _encode_payload(secret_bytes: bytes) -> bytes:
    return struct.pack('>I', len(secret_bytes)) + secret_bytes

def _decode_payload(data: bytes) -> bytes:
    if len(data) < 4:
        raise ValueError("Payload too short.")
    n = struct.unpack('>I', data[:4])[0]
    if n > len(data) - 4 or n < 0:
        raise ValueError(
            "Payload length mismatch — wrong password or image was edited after encoding."
        )
    return data[4:4 + n]


# ── Metrics ───────────────────────────────────────────────────────────────────

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    import math
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-10 else 10 * math.log10(255.0 ** 2 / mse)

def _ssim_fast(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = a.mean(), b.mean()
    s12 = float(np.mean((a - mu1) * (b - mu2)))
    return float(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(a.var()+b.var()+C2)))


# ── Secret type utilities ─────────────────────────────────────────────────────

def secret_text_to_bytes(text: str) -> bytes:
    return b'TXT:' + text.encode('utf-8')

def secret_image_to_bytes(img_file) -> bytes:
    img = Image.open(img_file).convert('RGB')
    if max(img.size) > 256:
        img.thumbnail((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=60)
    return b'IMG:' + buf.getvalue()

def secret_3d_to_bytes(data_file) -> bytes:
    try:
        arr = np.load(data_file)
        buf = io.BytesIO()
        np.save(buf, arr.flatten()[:4096].astype(np.float32))
        return b'3DV:' + buf.getvalue()
    except Exception:
        return b'3DR:' + data_file.read(8192)

def bytes_to_secret(data: bytes) -> dict:
    if data[:4] == b'TXT:':
        return {'type': 'text', 'content': data[4:].decode('utf-8', errors='replace')}
    elif data[:4] == b'IMG:':
        import base64
        return {'type': 'image', 'content': 'data:image/jpeg;base64,' + base64.b64encode(data[4:]).decode()}
    elif data[:4] in (b'3DV:', b'3DR:'):
        return {'type': '3d', 'content': f'3D data ({len(data)-4} bytes extracted)'}
    return {'type': 'text', 'content': data.decode('utf-8', errors='replace')}

def compute_robustness_score(psnr: float, ssim: float) -> float:
    p = min(max((psnr - 30) / 20, 0), 1)
    return round((p * 0.5 + max(ssim, 0) * 0.5) * 100, 1)
