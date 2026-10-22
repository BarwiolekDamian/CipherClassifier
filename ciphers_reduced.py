"""
ciphers_reduced.py
Educational pure Python/NumPy implementation of reduced-round block ciphers:
- AES-reduced (1 or 2 rounds, 128-bit / 16 B block size, ECB mode)
- DES-reduced (reduced-round Feistel network, e.g. 2 rounds, 64-bit / 8 B block size, ECB mode)
- 3DES-reduced (Triple DES EDE with reduced rounds, 64-bit / 8 B block size, ECB mode)
"""

import numpy as np

# =====================================================================
# AES REDUCED IMPLEMENTATION
# =====================================================================

AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5e, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

AES_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]


def _xtime(a: int) -> int:
    """Multiplication by 2 in GF(2^8) modulo irreducible polynomial x^8 + x^4 + x^3 + x + 1."""
    return ((a << 1) ^ 0x1b) & 0xff if (a & 0x80) else (a << 1) & 0xff


def _mix_single_column(a: np.ndarray) -> np.ndarray:
    """Mixes a single 4-byte column vector in Galois field GF(2^8)."""
    t = int(a[0]) ^ int(a[1]) ^ int(a[2]) ^ int(a[3])
    u = int(a[0])
    r0 = a[0] ^ _xtime(a[0] ^ a[1]) ^ t
    r1 = a[1] ^ _xtime(a[1] ^ a[2]) ^ t
    r2 = a[2] ^ _xtime(a[2] ^ a[3]) ^ t
    r3 = a[3] ^ _xtime(u ^ a[3]) ^ t
    return np.array([r0, r1, r2, r3], dtype=np.uint8)


def _aes_key_expansion_128(key: bytes, rounds: int = 2) -> list[np.ndarray]:
    """Generates round subkeys for AES-128."""
    w = [list(key[4 * i : 4 * (i + 1)]) for i in range(4)]
    for i in range(4, 4 * (rounds + 1)):
        temp = list(w[i - 1])
        if i % 4 == 0:
            # RotWord, SubWord and XOR with Rcon
            temp = temp[1:] + temp[:1]
            temp = [int(AES_SBOX[b]) for b in temp]
            temp[0] ^= AES_RCON[(i // 4) - 1]
        word = [w[i - 4][j] ^ temp[j] for j in range(4)]
        w.append(word)

    round_keys = []
    for r in range(rounds + 1):
        rk_bytes = []
        for col in range(4):
            rk_bytes.extend(w[r * 4 + col])
        round_keys.append(np.array(rk_bytes, dtype=np.uint8).reshape((4, 4)).T)
    return round_keys


def encrypt_aes_block_reduced(block: np.ndarray, round_keys: list[np.ndarray], rounds: int = 1) -> np.ndarray:
    """Encrypts a single 16-byte AES block for the specified number of rounds."""
    # State matrix in column-major order
    state = block.reshape((4, 4)).T.copy()
    state ^= round_keys[0]  # Initial round key addition

    for r in range(1, rounds + 1):
        # SubBytes substitution
        state = AES_SBOX[state]
        # ShiftRows transposition
        for i in range(4):
            state[i] = np.roll(state[i], -i)
        # MixColumns diffusion
        for col in range(4):
            state[:, col] = _mix_single_column(state[:, col])
        # Round key addition
        state ^= round_keys[r]

    return state.T.flatten()


def encrypt_aes_reduced(plaintext: bytes, key: bytes, rounds: int = 1) -> bytes:
    """Encrypts a byte sequence in ECB mode using reduced-round AES."""
    round_keys = _aes_key_expansion_128(key, rounds=rounds)
    arr = np.frombuffer(plaintext, dtype=np.uint8)
    out = np.empty_like(arr)
    num_blocks = len(arr) // 16
    for i in range(num_blocks):
        blk = arr[i * 16 : (i + 1) * 16]
        out[i * 16 : (i + 1) * 16] = encrypt_aes_block_reduced(blk, round_keys, rounds=rounds)
    return out.tobytes()


# =====================================================================
# DES REDUCED IMPLEMENTATION
# =====================================================================

DES_IP = [
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7
]

DES_FP = [
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25
]

DES_E = [
    32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1
]

DES_P = [
    16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25
]

DES_SBOXES = [
    # S1
    [
        14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
        0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
        4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
        15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13
    ],
    # S2
    [
        15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
        3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
        0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
        13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9
    ],
    # S3
    [
        10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
        13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
        13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
        1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12
    ],
    # S4
    [
        7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
        13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
        10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
        3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14
    ],
    # S5
    [
        2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
        14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
        4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
        11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3
    ],
    # S6
    [
        12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
        10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
        9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
        4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13
    ],
    # S7
    [
        4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
        13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
        1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
        6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12
    ],
    # S8
    [
        13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
        1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
        7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
        2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11
    ]
]

DES_PC1 = [
    57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4
]

DES_PC2 = [
    14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32
]

DES_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


def _permute(bits: np.ndarray, table: list[int]) -> np.ndarray:
    """Permutes a bit array according to a 1-indexed permutation table."""
    return bits[np.array(table) - 1]


def _des_key_schedule(key_bytes: bytes, rounds: int = 2) -> list[np.ndarray]:
    """Generates 48-bit round subkeys for reduced rounds of DES."""
    key_bits = np.unpackbits(np.frombuffer(key_bytes[:8], dtype=np.uint8))
    pc1_bits = _permute(key_bits, DES_PC1)
    c, d = pc1_bits[:28], pc1_bits[28:]
    subkeys = []
    for r in range(rounds):
        shift = DES_SHIFTS[r]
        c = np.roll(c, -shift)
        d = np.roll(d, -shift)
        cd = np.concatenate([c, d])
        subkeys.append(_permute(cd, DES_PC2))
    return subkeys


def _des_feistel_f(r_bits: np.ndarray, subkey_48: np.ndarray) -> np.ndarray:
    """The Feistel round function F in DES."""
    # Expansion from 32 to 48 bits
    expanded = _permute(r_bits, DES_E)
    # Key mixing with round subkey
    xored = expanded ^ subkey_48
    # S-Box substitution
    sbox_out = []
    for s_idx in range(8):
        chunk = xored[s_idx * 6 : (s_idx + 1) * 6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        val = DES_SBOXES[s_idx][row * 16 + col]
        sbox_out.extend([(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1])
    # Permutation P
    return _permute(np.array(sbox_out, dtype=np.uint8), DES_P)


def encrypt_des_block_reduced(block_bytes: bytes, subkeys: list[np.ndarray]) -> bytes:
    """Encrypts a single 8-byte block through reduced Feistel rounds."""
    bits = np.unpackbits(np.frombuffer(block_bytes, dtype=np.uint8))
    ip_bits = _permute(bits, DES_IP)
    l, r = ip_bits[:32], ip_bits[32:]

    for k in subkeys:
        l_next = r
        r_next = l ^ _des_feistel_f(r, k)
        l, r = l_next, r_next

    # Swap left and right halves before final permutation
    pre_fp = np.concatenate([r, l])
    cipher_bits = _permute(pre_fp, DES_FP)
    return np.packbits(cipher_bits).tobytes()


def encrypt_des_reduced(plaintext: bytes, key: bytes, rounds: int = 2) -> bytes:
    """Encrypts a byte sequence in ECB mode using reduced-round DES."""
    subkeys = _des_key_schedule(key, rounds=rounds)
    out = bytearray(len(plaintext))
    num_blocks = len(plaintext) // 8
    for i in range(num_blocks):
        blk = plaintext[i * 8 : (i + 1) * 8]
        out[i * 8 : (i + 1) * 8] = encrypt_des_block_reduced(blk, subkeys)
    return bytes(out)


# =====================================================================
# 3DES REDUCED IMPLEMENTATION
# =====================================================================

def encrypt_3des_reduced(plaintext: bytes, key: bytes, rounds_per_stage: int = 1) -> bytes:
    """
    Reduced Triple DES in EDE (Encrypt-Decrypt-Encrypt) mode:
    Accepts 24-byte key (three 8-byte subkeys).
    Each phase executes reduced DES Feistel rounds.
    """
    if len(key) < 24:
        key = (key * 3)[:24]
    k1, k2, k3 = key[:8], key[8:16], key[16:24]

    # Encrypt using first subkey
    step1 = encrypt_des_reduced(plaintext, k1, rounds=rounds_per_stage)
    # Decrypt using second subkey with reversed schedule
    subkeys_k2 = _des_key_schedule(k2, rounds=rounds_per_stage)[::-1]
    step2 = bytearray(len(step1))
    for i in range(len(step1) // 8):
        blk = step1[i * 8 : (i + 1) * 8]
        step2[i * 8 : (i + 1) * 8] = encrypt_des_block_reduced(blk, subkeys_k2)
    # Encrypt using third subkey
    step3 = encrypt_des_reduced(bytes(step2), k3, rounds=rounds_per_stage)
    return step3


if __name__ == "__main__":
    print("==================================================")
    print(" RUNNING REDUCED CIPHERS VERIFICATION SUITE       ")
    print("==================================================")

    sample_plaintext = b"A" * 1024
    key16 = b"\x01\x23\x45\x67\x89\xab\xcd\xef\xfe\xdc\xba\x98\x76\x54\x32\x10"
    key24 = key16 + b"\x55\xaa\x55\xaa\x55\xaa\x55\xaa"

    print(f"[*] Input sample size: {len(sample_plaintext)} bytes")

    print("\n--- Testing AES-reduced ---")
    aes_out = encrypt_aes_reduced(sample_plaintext, key16, rounds=1)
    print(f"  Output size : {len(aes_out)} bytes")
    print(f"  First 16 B  : {aes_out[:16].hex()}")
    assert len(aes_out) == 1024, "AES output must match exact 1024 bytes"
    print("  Status      : [PASSED]")

    print("\n--- Testing DES-reduced ---")
    des_out = encrypt_des_reduced(sample_plaintext, key16[:8], rounds=2)
    print(f"  Output size : {len(des_out)} bytes")
    print(f"  First 16 B  : {des_out[:16].hex()}")
    assert len(des_out) == 1024, "DES output must match exact 1024 bytes"
    print("  Status      : [PASSED]")

    print("\n--- Testing 3DES-reduced ---")
    tdes_out = encrypt_3des_reduced(sample_plaintext, key24, rounds_per_stage=1)
    print(f"  Output size : {len(tdes_out)} bytes")
    print(f"  First 16 B  : {tdes_out[:16].hex()}")
    assert len(tdes_out) == 1024, "3DES output must match exact 1024 bytes"
    print("  Status      : [PASSED]")

    print("\n==================================================")
    print(" ALL TESTS PASSED: 1024 B outputs verified!       ")
    print("==================================================")
