from PIL import Image
import numpy as np
import torch
from utils.neural_stego import (
    _pw_seed, _attention_map, _sorted_positions, _int_to_bits, _bits_to_int,
    _MAGIC, neural_embed, secret_text_to_bytes
)

pw = 'NeuralAI2024'
seed = _pw_seed(pw)

# Create test cover
arr = np.random.RandomState(42).randint(50, 200, (300, 300, 3), dtype=np.uint8)
img = Image.fromarray(arr, 'RGB')

print("=== DEBUG NEURAL STEGO ===")
print(f"MAGIC expected: {hex(_MAGIC)}")
print(f"Password seed:  {hex(seed)}")

# Embed
secret = secret_text_to_bytes('Test')
stego, psnr, ssim = neural_embed(img, secret, pw)
stego_arr = np.array(stego, dtype=np.uint8)

# Replicate embed-side attention
att_embed = _attention_map(arr, pw)
print(f"Att embed shape: {att_embed.shape}, min={att_embed.min():.4f} max={att_embed.max():.4f}")

# Replicate extract-side attention
att_extract = _attention_map(stego_arr, pw)
print(f"Att extract shape: {att_extract.shape}, min={att_extract.min():.4f} max={att_extract.max():.4f}")

# Check if attention maps are identical
diff = np.abs(att_embed - att_extract).max()
print(f"Max att difference: {diff}")

# Get embed positions for header copy 0
hdr_pos_embed = _sorted_positions(att_embed, 64, seed ^ 0xAAAA, 0)
hdr_pos_extract = _sorted_positions(att_extract, 64, seed ^ 0xAAAA, 0)
print(f"Positions match: {np.array_equal(hdr_pos_embed, hdr_pos_extract)}")
print(f"First 5 embed positions:   {hdr_pos_embed[:5]}")
print(f"First 5 extract positions: {hdr_pos_extract[:5]}")

# Check what bits are at those positions
flat_stego = stego_arr.flatten().astype(np.int32)
magic_bits = [int(flat_stego[int(hdr_pos_extract[i])]) & 1 for i in range(32)]
read_magic = _bits_to_int(magic_bits)
expected_magic_bits = _int_to_bits(_MAGIC, 32)

print(f"Expected MAGIC bits: {expected_magic_bits[:8]}...")
print(f"Read    MAGIC bits:  {magic_bits[:8]}...")
print(f"Expected MAGIC: {hex(_MAGIC)}")
print(f"Read    MAGIC:  {hex(read_magic)}")
print(f"Match: {read_magic == _MAGIC}")
