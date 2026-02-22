from utils.cover_3d import embed_in_3d, extract_from_3d
from utils.neural_stego import bytes_to_secret, secret_text_to_bytes
import numpy as np, io

print("=== 3D COVER TESTS ===")

# Test 1: OBJ binary cover
obj = (b'v 1.0 2.0 3.0\nv 4.0 5.0 6.0\nf 1 2 3\n') * 500
secret = secret_text_to_bytes('Hidden inside a 3D model!')
stego = embed_in_3d(obj, secret, 'pass123', 'model.obj')
print(f"OBJ embed: original={len(obj)} bytes, stego={len(stego)} bytes")
raw = extract_from_3d(stego, 'pass123', 'model.obj')
result = bytes_to_secret(raw)
print(f"OBJ extract: {result}")
assert result['content'] == 'Hidden inside a 3D model!'
print("[OK] OBJ Cover")

# Test 2: NPY cover
arr = np.random.rand(100, 100).astype(np.float32)
buf = io.BytesIO()
np.save(buf, arr)
npy_bytes = buf.getvalue()
stego2 = embed_in_3d(npy_bytes, secret, 'pass456', 'data.npy')
print(f"NPY embed: original={len(npy_bytes)} bytes, stego={len(stego2)} bytes")
raw2 = extract_from_3d(stego2, 'pass456', 'data.npy')
result2 = bytes_to_secret(raw2)
print(f"NPY extract: {result2}")
assert result2['content'] == 'Hidden inside a 3D model!'
print("[OK] NPY Cover")

# Test 3: BIN cover
bin_data = bytes(range(256)) * 400
stego3 = embed_in_3d(bin_data, secret, 'pass789', 'data.bin')
raw3 = extract_from_3d(stego3, 'pass789', 'data.bin')
result3 = bytes_to_secret(raw3)
assert result3['content'] == 'Hidden inside a 3D model!'
print("[OK] BIN Cover")

print("\nALL 3D COVER TESTS PASSED!")
