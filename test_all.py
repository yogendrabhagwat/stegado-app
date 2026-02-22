import sys, io
import numpy as np
from PIL import Image
from utils.neural_stego import (
    neural_embed, neural_extract,
    secret_text_to_bytes, secret_3d_to_bytes,
    bytes_to_secret, compute_robustness_score
)

print("=== STEGADO COMPREHENSIVE TEST ===")
errors = []

arr = np.random.randint(50, 200, (600, 600, 3), dtype=np.uint8)
img = Image.fromarray(arr, 'RGB')

# 1. Text hide/extract
try:
    secret = secret_text_to_bytes('Stegado Neural AI Test 2024!')
    stego, psnr, ssim = neural_embed(img, secret, 'TestPwd2024')
    raw = neural_extract(stego, 'TestPwd2024')
    result = bytes_to_secret(raw)
    assert result['type'] == 'text'
    assert result['content'] == 'Stegado Neural AI Test 2024!'
    rob = compute_robustness_score(psnr, ssim)
    print(f"[OK] Text | PSNR={psnr} dB | SSIM={ssim} | Robustness={rob}%")
except Exception as e:
    errors.append(f"Text: {e}")
    print(f"[FAIL] Text: {e}")
    stego = None

# 2. Wrong password blocked
try:
    if stego:
        try:
            neural_extract(stego, 'wrongpassword')
            errors.append("Wrong password NOT blocked!")
            print("[FAIL] Wrong password not blocked")
        except ValueError:
            print("[OK] Wrong password blocked correctly")
except Exception as e:
    errors.append(f"Password: {e}")

# 3. 3D data
try:
    class FakeFile:
        def read(self, n=-1): return b'3D_TEST_DATA_XYZ_' * 200
    secret_3d = secret_3d_to_bytes(FakeFile())
    stego2, p2, s2 = neural_embed(img, secret_3d, 'pwd3D2024')
    raw2 = neural_extract(stego2, 'pwd3D2024')
    r2 = bytes_to_secret(raw2)
    assert r2['type'] == '3d'
    assert 'data' in r2, "No base64 data for download!"
    assert 'ext' in r2
    print(f"[OK] 3D data | type={r2['type']} | {r2['content']} | download ready")
except Exception as e:
    errors.append(f"3D: {e}")
    print(f"[FAIL] 3D: {e}")

# 4. PNG round-trip (save PNG → reload → extract)
try:
    if stego:
        buf = io.BytesIO()
        stego.save(buf, format='PNG')
        buf.seek(0)
        reloaded = Image.open(buf)
        raw_r = neural_extract(reloaded, 'TestPwd2024')
        result_r = bytes_to_secret(raw_r)
        assert result_r['content'] == 'Stegado Neural AI Test 2024!'
        print("[OK] PNG save/reload roundtrip")
except Exception as e:
    errors.append(f"PNG roundtrip: {e}")
    print(f"[FAIL] PNG roundtrip: {e}")

# 5. Flask app
try:
    import os
    os.environ.setdefault('SECRET_KEY', 'test')
    from app import create_app
    app = create_app()
    print("[OK] Flask app creates successfully")
except Exception as e:
    errors.append(f"Flask: {e}")
    print(f"[FAIL] Flask: {e}")

print()
if errors:
    print(f"FAILURES: {len(errors)}")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED! Stegado is 100% working!")
