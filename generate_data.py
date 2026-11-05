"""
generate_data.py
Phase 1: Dataset generation pipeline for Ciphertext-Only Attack (COA) classifier.

Generates:
- Balanced 4-class training/evaluation dataset (AES-reduced, DES-reduced, 3DES-reduced, RC4)
- Adversarial evaluation dataset on full-round ciphers (AES-10r, DES-16r)
- Saves outputs as compressed .npz archives and individual .npy matrices for team compatibility
"""

import argparse
import os
import urllib.request
import numpy as np
from Crypto.Cipher import AES, DES

from ciphers_reduced import (
    encrypt_aes_reduced,
    encrypt_des_reduced,
    encrypt_3des_reduced,
    encrypt_rc4
)

BOOK_URLS = [
    ("lalka_1.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"),
    ("lalka_2.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-drugi.txt"),
    ("quo_vadis.txt", "https://wolnelektury.pl/media/book/txt/quo-vadis.txt"),
    ("faraon_1.txt", "https://wolnelektury.pl/media/book/txt/faraon-tom-pierwszy.txt"),
]


def download_plaintext(url: str, filepath: str) -> bytes:
    """Downloads text from public domain archive if not cached locally."""
    if not os.path.exists(filepath):
        print(f"[*] Downloading text from: {url} -> {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("[*] Download completed.")
        except Exception as e:
            print(f"[!] Warning: Could not download {url}: {e}")
            return b""

    with open(filepath, "rb") as f:
        data = f.read()

    print(f"[*] Loaded {len(data) / 1024:.2f} KB of raw text from {filepath}")
    return data


def load_all_plaintexts(raw_dir: str = "data/raw") -> bytes:
    """Loads and aggregates plaintexts from local cache or downloads them."""
    os.makedirs(raw_dir, exist_ok=True)
    all_bytes = bytearray()
    for filename, url in BOOK_URLS:
        filepath = os.path.join(raw_dir, filename)
        content = download_plaintext(url, filepath)
        all_bytes.extend(content)
    return bytes(all_bytes)


def prepare_blocks(data_bytes: bytes, block_size: int = 1024) -> list[bytes]:
    """Slices natural text into non-overlapping blocks of exact block_size bytes."""
    blocks = []
    for i in range(0, len(data_bytes) - block_size + 1, block_size):
        blocks.append(data_bytes[i : i + block_size])
    print(f"[*] Prepared {len(blocks)} unique natural text blocks ({block_size} bytes each).")
    return blocks


def encrypt_aes_full(block: bytes) -> bytes:
    """Encrypts 1024-byte block using full 10-round AES in ECB mode."""
    key = os.urandom(16)
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(block)


def encrypt_des_full(block: bytes) -> bytes:
    """Encrypts 1024-byte block using full 16-round DES in ECB mode."""
    key = os.urandom(8)
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(block)


def generate_main_dataset(text_blocks: list[bytes], samples_per_class: int = 5000, aes_rounds: int = 1, des_rounds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Generates balanced 4-class dataset using reduced-round ciphers and independent random keys."""
    print(f"\n[*] Generating main dataset: 4 classes x {samples_per_class} samples (total {4 * samples_per_class})...")
    X_list = []
    y_list = []

    total_blocks = len(text_blocks)
    block_idx = 0

    for class_id in range(4):
        for _ in range(samples_per_class):
            block = text_blocks[block_idx % total_blocks]
            block_idx += 1

            if class_id == 0:
                encrypted = encrypt_aes_reduced(block, rounds=aes_rounds)
            elif class_id == 1:
                encrypted = encrypt_des_reduced(block, rounds=des_rounds)
            elif class_id == 2:
                encrypted = encrypt_3des_reduced(block, rounds_per_stage=1)
            elif class_id == 3:
                encrypted = encrypt_rc4(block)

            sample_array = np.frombuffer(encrypted, dtype=np.uint8)
            X_list.append(sample_array)
            y_list.append(class_id)

    X = np.array(X_list, dtype=np.uint8)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def generate_adversarial_dataset(text_blocks: list[bytes], samples_per_cipher: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generates adversarial evaluation set using full-round ciphers (AES-10r labeled 0, DES-16r labeled 1)."""
    print(f"\n[*] Generating adversarial dataset: {samples_per_cipher} AES-10r + {samples_per_cipher} DES-16r...")
    X_list = []
    y_list = []

    total_blocks = len(text_blocks)
    block_idx = 0

    for _ in range(samples_per_cipher):
        block = text_blocks[block_idx % total_blocks]
        block_idx += 1
        encrypted = encrypt_aes_full(block)
        X_list.append(np.frombuffer(encrypted, dtype=np.uint8))
        y_list.append(0)

    for _ in range(samples_per_cipher):
        block = text_blocks[block_idx % total_blocks]
        block_idx += 1
        encrypted = encrypt_des_full(block)
        X_list.append(np.frombuffer(encrypted, dtype=np.uint8))
        y_list.append(1)

    X_adv = np.array(X_list, dtype=np.uint8)
    y_adv = np.array(y_list, dtype=np.int64)
    return X_adv, y_adv


def main():
    parser = argparse.ArgumentParser(description="Generate COA cipher datasets (Phase 1)")
    parser.add_argument("--samples-per-class", type=int, default=1500,
                        help="Number of samples per class for main dataset (default: 1500, target: 5000)")
    parser.add_argument("--adv-samples", type=int, default=500,
                        help="Number of adversarial samples per cipher (default: 500, target: 1000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with small dataset (350 samples/class, 150 adv)")
    parser.add_argument("--full", action="store_true",
                        help="Generate full target dataset (5000 samples/class, 1000 adv)")
    parser.add_argument("--aes-rounds", type=int, default=1,
                        help="Rounds for AES-reduced (default: 1)")
    parser.add_argument("--des-rounds", type=int, default=2,
                        help="Rounds for DES-reduced (default: 2)")
    parser.add_argument("--output-dir", type=str, default="data/ciphertexts",
                        help="Directory to save generated datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory for raw text downloads")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.full:
        samples_per_class = 5000
        adv_samples = 1000
    elif args.quick:
        samples_per_class = 350
        adv_samples = 150
    else:
        samples_per_class = args.samples_per_class
        adv_samples = args.adv_samples

    print("==================================================")
    print(" PHASE 1: DATASET GENERATION PIPELINE             ")
    print("==================================================")
    print(f"[*] Configuration: samples_per_class={samples_per_class}, adv_samples={adv_samples}")

    raw_bytes = load_all_plaintexts(args.raw_dir)
    text_blocks = prepare_blocks(raw_bytes, block_size=1024)

    if not text_blocks:
        raise ValueError("No text blocks available for encryption.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Main dataset generation
    X_main, y_main = generate_main_dataset(
        text_blocks,
        samples_per_class=samples_per_class,
        aes_rounds=args.aes_rounds,
        des_rounds=args.des_rounds
    )
    
    # Save as .npz archive for extract_features pipeline
    main_npz_path = os.path.join(args.output_dir, "dataset.npz")
    np.savez_compressed(main_npz_path, X=X_main, y=y_main)

    # Save as separate .npy matrices for compatibility with teammate's branch
    np.save(os.path.join(args.output_dir, "X_ciphertexts.npy"), X_main)
    np.save(os.path.join(args.output_dir, "y_labels.npy"), y_main)
    print(f"[✓] Saved main dataset to {main_npz_path} (shape: {X_main.shape})")

    # Adversarial dataset generation
    X_adv, y_adv = generate_adversarial_dataset(text_blocks, samples_per_cipher=adv_samples)
    adv_npz_path = os.path.join(args.output_dir, "adversarial.npz")
    np.savez_compressed(adv_npz_path, X=X_adv, y=y_adv)
    print(f"[✓] Saved adversarial dataset to {adv_npz_path} (shape: {X_adv.shape})")

    print("\n==================================================")
    print(" DATASET GENERATION COMPLETED SUCCESSFULLY!       ")
    print("==================================================")


if __name__ == "__main__":
    main()
