# TODO: Implement dataset generation pipeline (Phase 1):
# - Load low-entropy natural plaintext (e.g., natural language texts from data/raw/)
# - Slice into fixed-size 1024-byte (1 KB) blocks without variable padding
# - Encrypt with independent random keys per sample:
#     * Class 0: AES-reduced (1-2 rounds)
#     * Class 1: DES-reduced (2 rounds)
#     * Class 2: 3DES-reduced (2-3 rounds)
#     * Class 3: RC4 (stream cipher via pycryptodome)
# - Generate 5,000 samples per class (total 20,000 samples)
# - Generate adversarial evaluation set with full rounds (min. 1,000 AES-10r and 1,000 DES-16r)
# - Save raw arrays to data/ciphertexts/
