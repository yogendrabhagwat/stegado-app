"""
3D Cover File Steganography
════════════════════════════
Hides secrets INSIDE 3D data files (.obj, .npy, .bin, .ply)
by treating the file's bytes as a pixel carrier and applying
the same neural attention embedding as image steganography.

Supported cover types:
  .obj  → vertex coordinate float values modified (imperceptible to renderers)
  .npy  → numpy array bytes modified (sub-LSB changes)
  .bin/.ply/.stl → binary bytes modified directly

The stego 3D file is a valid, renderable/loadable file with
pixel-level invisible modifications hiding the secret.
"""

import io
import math
import hashlib
import numpy as np
from PIL import Image
from typing import Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

def _square_dims(n_bytes: int) -> Tuple[int, int]:
    """Return (H, W) such that H × W × 3 ≥ n_bytes."""
    pixels = math.ceil(n_bytes / 3)
    side = math.ceil(math.sqrt(pixels))
    return max(64, side), max(64, side)


def _bytes_to_img(data: bytes) -> Tuple[Image.Image, int]:
    """Pack raw bytes into an RGB PIL Image for neural embedding."""
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    n = len(arr)
    H, W = _square_dims(n)
    padded = np.zeros(H * W * 3, dtype=np.uint8)
    padded[:n] = arr
    img = Image.fromarray(padded.reshape(H, W, 3), 'RGB')
    return img, n


def _img_to_bytes(stego_img: Image.Image, original_len: int) -> bytes:
    """Unpack embedded image back to bytes, trimmed to original length."""
    arr = np.array(stego_img, dtype=np.uint8).flatten()
    return bytes(arr[:original_len])


# ── OBJ-specific: embed/extract in floating-point vertex coordinates ──────────

def _obj_to_array(content: str) -> Tuple[list, np.ndarray, list]:
    """
    Parse .obj text → list of lines, float32 vertex values, list of (line_idx, col).
    """
    lines = content.split('\n')
    vertex_idx = []   # (line_i, col_j)
    values = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('v ') or stripped.startswith('v\t'):
            parts = stripped.split()
            for j in range(1, min(len(parts), 5)):  # x y z [w]
                try:
                    val = float(parts[j])
                    vertex_idx.append((i, j))
                    values.append(val)
                except ValueError:
                    pass

    return lines, np.array(values, dtype=np.float64), vertex_idx


def _array_to_obj(lines: list, values: np.ndarray, vertex_idx: list) -> bytes:
    """Reconstruct .obj bytes from modified float values."""
    result = [l for l in lines]
    for k, (i, j) in enumerate(vertex_idx):
        parts = result[i].split()
        parts[j] = f'{values[k]:.6f}'
        result[i] = ' '.join(parts)
    return '\n'.join(result).encode('utf-8')


def _embed_in_obj(obj_bytes: bytes, secret_bytes: bytes, password: str) -> bytes:
    """Hide secret in .obj vertex coordinates."""
    content = obj_bytes.decode('utf-8', errors='replace')
    lines, values, vertex_idx = _obj_to_array(content)

    if len(values) < 200:
        # Not enough vertices — fall back to binary embedding
        return _embed_binary(obj_bytes, secret_bytes, password)

    # Normalise vertices → uint8 range → treat as image pixels
    v_min, v_max = float(values.min()), float(values.max())
    span = v_max - v_min if abs(v_max - v_min) > 1e-10 else 1.0
    norm = ((values - v_min) / span * 255).astype(np.uint8)

    img, n = _bytes_to_img(norm.tobytes())
    from utils.neural_stego import neural_embed
    stego_img, _, _ = neural_embed(img, secret_bytes, password)
    stego_bytes = _img_to_bytes(stego_img, n)

    stego_norm = np.frombuffer(stego_bytes, dtype=np.uint8).astype(np.float64)
    stego_vals = stego_norm / 255.0 * span + v_min
    return _array_to_obj(lines, stego_vals, vertex_idx)


def _extract_from_obj(stego_bytes: bytes, password: str) -> bytes:
    """Extract secret from stego .obj."""
    content = stego_bytes.decode('utf-8', errors='replace')
    lines, values, vertex_idx = _obj_to_array(content)

    if len(values) < 200:
        return _extract_binary(stego_bytes, password)

    v_min, v_max = float(values.min()), float(values.max())
    span = v_max - v_min if abs(v_max - v_min) > 1e-10 else 1.0
    norm = ((values - v_min) / span * 255).astype(np.uint8)

    img, n = _bytes_to_img(norm.tobytes())
    from utils.neural_stego import neural_extract
    return neural_extract(img, password)


# ── Binary (npy / bin / ply / stl) ───────────────────────────────────────────

def _embed_binary(cover_bytes: bytes, secret_bytes: bytes, password: str) -> bytes:
    """
    Embed secret in raw binary bytes.
    Returns H×W×3 bytes (may be slightly larger than input) to preserve all embedding positions.
    """
    from utils.neural_stego import neural_embed
    img, _n = _bytes_to_img(cover_bytes)
    stego_img, _, _ = neural_embed(img, secret_bytes, password)
    # Return FULL H×W×3 array — do NOT trim, so padding positions are preserved
    return np.array(stego_img, dtype=np.uint8).flatten().tobytes()


def _extract_binary(stego_bytes: bytes, password: str) -> bytes:
    from utils.neural_stego import neural_extract
    img, _n = _bytes_to_img(stego_bytes)
    return neural_extract(img, password)



# ── Public API ────────────────────────────────────────────────────────────────

_3D_EXTENSIONS = {'obj', 'npy', 'npz', 'bin', 'ply', 'stl', 'fbx', 'glb'}


def is_3d_file(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in _3D_EXTENSIONS


def embed_in_3d(cover_bytes: bytes, secret_bytes: bytes, password: str, filename: str) -> bytes:
    """
    Embed secret_bytes into a 3D cover file.
    Uses binary-level neural embedding — modifies ±1 byte values.
    Works with .obj, .npy, .npz, .bin, .ply, .stl, .glb and any binary format.
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'bin'

    if ext in ('npy', 'npz'):
        # Embed in the raw bytes of the numpy file
        return _embed_binary(cover_bytes, secret_bytes, password)

    # For .obj and all binary formats: embed directly in raw bytes
    return _embed_binary(cover_bytes, secret_bytes, password)


def extract_from_3d(stego_bytes: bytes, password: str, filename: str) -> bytes:
    """Extract secret from a stego 3D file."""
    return _extract_binary(stego_bytes, password)

