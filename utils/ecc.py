"""
Reed-Solomon Error Correction Coding for robustness.
Uses 32 error correction symbols – can recover from up to 16 byte errors per block.
"""
import reedsolo

RS_NSYM = 32          # ECC symbols (error correction power = 16 bytes)
BLOCK_SIZE = 200      # Process data in chunks to handle large payloads


def encode_ecc(data: bytes) -> bytes:
    """Apply Reed-Solomon ECC to data bytes."""
    rs = reedsolo.RSCodec(RS_NSYM)
    # Process in blocks for large data
    encoded_blocks = []
    for i in range(0, len(data), BLOCK_SIZE):
        block = data[i:i + BLOCK_SIZE]
        encoded_blocks.append(bytes(rs.encode(block)))
    # Prepend block count (4 bytes) so decoder knows structure
    import struct
    header = struct.pack('>II', len(encoded_blocks), len(data))
    return header + b''.join(encoded_blocks)


def decode_ecc(data: bytes) -> bytes:
    """Decode and error-correct Reed-Solomon encoded data."""
    import struct
    if len(data) < 8:
        raise ValueError("ECC data too short.")
    num_blocks, original_len = struct.unpack('>II', data[:8])
    data = data[8:]

    rs = reedsolo.RSCodec(RS_NSYM)
    block_enc_size = BLOCK_SIZE + RS_NSYM
    decoded_parts = []

    for i in range(num_blocks):
        block = data[i * block_enc_size:(i + 1) * block_enc_size]
        if not block:
            break
        try:
            decoded_block, _, _ = rs.decode(block)
            decoded_parts.append(bytes(decoded_block))
        except reedsolo.ReedSolomonError:
            # Too many errors – append zeros, let AES GCM auth catch it
            decoded_parts.append(b'\x00' * BLOCK_SIZE)

    result = b''.join(decoded_parts)
    return result[:original_len]
