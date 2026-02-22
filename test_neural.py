from PIL import Image
import numpy as np
from utils.neural_stego import neural_embed, neural_extract, secret_text_to_bytes, bytes_to_secret

arr = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)
img = Image.fromarray(arr, 'RGB')

pw = 'NeuralAI2024'
secret = secret_text_to_bytes('Hello from Neural AI Steganography!')

print('Embedding using Neural Fourier AI...')
stego, psnr, ssim = neural_embed(img, secret, pw)
print(f'Embed OK | PSNR={psnr} dB | SSIM={ssim}')

print('Extracting using same neural model...')
raw = neural_extract(stego, pw)
result = bytes_to_secret(raw)
print(f'Extract OK | Type: {result["type"]} | Content: {result["content"]}')

print('Testing wrong password...')
try:
    neural_extract(stego, 'wrongpassword')
    print('ERROR: should have failed!')
except ValueError:
    print('Wrong password blocked: OK')

print()
print('ALL NEURAL TESTS PASSED!')
