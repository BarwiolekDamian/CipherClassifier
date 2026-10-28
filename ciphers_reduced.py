import os
from Crypto.Cipher import DES3, ARC4

def encrypt_aes_reduced(block: bytes, rounds: int = 2) -> bytes:
    key = os.urandom(16)
    state = bytearray(block)
    for r in range(rounds):
        for i in range(len(state)):
            state[i] = (state[i] ^ key[i % len(key)] + r) & 0xFF
    return bytes(state)

def encrypt_des_reduced(block: bytes, rounds: int = 2) -> bytes:
    key = os.urandom(8)
    state = bytearray(block)
    for r in range(rounds):
        for i in range(len(state)):
            state[i] = (state[i] ^ key[i % len(key)] ^ r) & 0xFF
    return bytes(state)

def encrypt_3des_reduced(block: bytes) -> bytes:
    key = DES3.adjust_key_parity(os.urandom(24))
    cipher = DES3.new(key, DES3.MODE_ECB)

    encrypted_blocks = bytearray()
    for i in range(0, len(block), 8):
        chunk = block[i:i + 8]
        if len(chunk) < 8:
            chunk = chunk + b'\x00' * (8 - len(chunk))
        encrypted_blocks.extend(cipher.encrypt(chunk))
    return bytes(encrypted_blocks)

def encrypt_rc4(block: bytes) -> bytes:
    key = os.urandom(16)
    cipher = ARC4.new(key)
    return cipher.encrypt(block)