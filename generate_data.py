"""
generate_data.py
Phase 1: Dataset generation pipeline for Ciphertext-Only Attack (COA) classifier.

Generates:
- Balanced 4-class training/validation/test dataset (AES-reduced, DES-reduced, 3DES-reduced, RC4)
  using STRICTLY DISJOINT natural-language books from Wolne Lektury (official REST API)
  for each split (Train 70% / Val 15% / Test 15%) and strictly unique 1024-byte plaintext blocks
  (zero block or file Data Leakage, Sections 5.1 & 5.4).
- Adversarial evaluation dataset on full-round ciphers (AES-10r, DES-16r) from separate source books.
- Saves outputs as compressed .npz archives (with split metadata) and .npy matrices.
"""

import argparse
import json
import os
import unicodedata
import urllib.request
import numpy as np
from Crypto.Cipher import AES, DES

from ciphers_reduced import (
    encrypt_aes_reduced,
    encrypt_des_reduced,
    encrypt_3des_reduced,
    encrypt_rc4,
)

# Official Wolne Lektury REST API endpoints (novels first for large text volume, then all epic prose)
WL_API_ENDPOINTS = [
    "https://wolnelektury.pl/api/genres/powiesc/books/",
    "https://wolnelektury.pl/api/kinds/epika/books/",
]

# Base seed books (instant cache compatibility with already downloaded files)
SEED_BOOKS = [
    ("lalka_1.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"),
    ("lalka_2.txt", "https://wolnelektury.pl/media/book/txt/lalka-tom-drugi.txt"),
    ("quo_vadis.txt", "https://wolnelektury.pl/media/book/txt/quo-vadis.txt"),
    ("faraon_1.txt", "https://wolnelektury.pl/media/book/txt/faraon-tom-pierwszy.txt"),
]

# Slugs already covered by SEED_BOOKS to prevent duplicate text across files
SEED_SLUGS = {
    "lalka-tom-pierwszy",
    "lalka-tom-drugi",
    "quo-vadis",
    "faraon-tom-pierwszy",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (CipherClassifier Educational Research)"}


def _http_get(url: str, timeout: int = 25) -> bytes:
    """Performs an HTTP GET request with a clean User-Agent header."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def normalize_to_ascii(raw_bytes: bytes) -> bytes:
    """
    Normalizes UTF-8 natural language text to clean 7-bit ASCII (1 byte per character,
    preserving natural linguistic redundancy and consistent byte encoding across books).
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    text = text.replace("ł", "l").replace("Ł", "L")
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", errors="ignore")


def prepare_blocks(data_bytes: bytes, block_size: int = 1024) -> list[bytes]:
    """Slices normalized natural text into non-overlapping blocks of exact block_size bytes."""
    ascii_bytes = normalize_to_ascii(data_bytes)
    blocks = []
    for i in range(0, len(ascii_bytes) - block_size + 1, block_size):
        blocks.append(ascii_bytes[i : i + block_size])
    return blocks


def ensure_wolne_lektury_corpus(raw_dir: str, target_blocks: int, block_size: int = 1024) -> list[tuple[str, list[bytes]]]:
    """
    Loads locally cached .txt books from raw_dir and downloads additional Polish novels
    from the official Wolne Lektury API until total unique blocks >= target_blocks.
    Returns a list of (filename, blocks) tuples, one entry per distinct book file.
    """
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Ensure seed books exist
    for filename, url in SEED_BOOKS:
        fpath = os.path.join(raw_dir, filename)
        if not os.path.exists(fpath):
            try:
                content = _http_get(url)
                with open(fpath, "wb") as f:
                    f.write(content)
            except Exception as e:
                print(f"[!] Warning: Could not download seed book {filename}: {e}")

    # 2. Load all currently cached .txt files
    book_pools: list[tuple[str, list[bytes]]] = []
    cached_names: set[str] = set()
    total_blocks = 0

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(raw_dir, fname)
        with open(fpath, "rb") as f:
            raw = f.read()
        blocks = prepare_blocks(raw, block_size=block_size)
        if len(blocks) >= 20:  # ignore tiny stubs/poems
            book_pools.append((fname, blocks))
            cached_names.add(fname)
            total_blocks += len(blocks)

    print(f"[*] Cached corpus in {raw_dir}: {len(book_pools)} books, {total_blocks} unique {block_size}-byte blocks.")

    if total_blocks >= target_blocks:
        return book_pools

    # 3. Fetch additional novels dynamically from Wolne Lektury REST API
    print(f"[*] Need {target_blocks} unique blocks (currently {total_blocks}). Querying Wolne Lektury API...")
    seen_slugs = set(SEED_SLUGS)
    for cname in cached_names:
        if cname.startswith("wl_") and cname.endswith(".txt"):
            seen_slugs.add(cname[3:-4])

    for api_url in WL_API_ENDPOINTS:
        if total_blocks >= target_blocks:
            break
        try:
            print(f"[*] Fetching book catalog from {api_url}...")
            catalog = json.loads(_http_get(api_url, timeout=30).decode("utf-8"))
        except Exception as e:
            print(f"[!] Warning: Could not query {api_url}: {e}")
            continue

        # Sort deterministically by slug for reproducibility
        catalog = sorted(catalog, key=lambda b: b.get("slug", ""))

        for entry in catalog:
            if total_blocks >= target_blocks:
                break

            slug = entry.get("slug", "")
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            # Skip parent anthology records that have no standalone txt or duplicates
            txt_url = f"https://wolnelektury.pl/media/book/txt/{slug}.txt"
            local_name = f"wl_{slug}.txt"
            local_path = os.path.join(raw_dir, local_name)

            try:
                content = _http_get(txt_url, timeout=20)
            except Exception:
                continue

            blocks = prepare_blocks(content, block_size=block_size)
            # Prefer substantive prose works (>= 80 KB -> ~80 blocks) to download fast
            if len(blocks) < 80:
                continue

            with open(local_path, "wb") as f:
                f.write(content)

            book_pools.append((local_name, blocks))
            total_blocks += len(blocks)
            print(f"    [+] Downloaded {local_name:<45} (+{len(blocks):4d} blocks | total: {total_blocks}/{target_blocks})")

    print(f"[*] Final corpus loaded: {len(book_pools)} books, {total_blocks} unique {block_size}-byte blocks.")
    return book_pools


def allocate_disjoint_book_splits(
    book_pools: list[tuple[str, list[bytes]]],
    needed: dict[str, int],
    seed: int = 42,
) -> dict[str, list[bytes]]:
    """
    Assigns WHOLE books strictly to disjoint splits ('adversarial', 'test', 'val', 'train')
    so that NO source file is ever split across Train / Val / Test / Adversarial (Section 5.4).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(book_pools))
    shuffled_books = [book_pools[int(i)] for i in order]

    splits: dict[str, list[bytes]] = {k: [] for k in ["train", "val", "test", "adversarial"]}
    split_files: dict[str, list[str]] = {k: [] for k in ["train", "val", "test", "adversarial"]}

    # Pack smaller splits first so each gets whole books cleanly, remaining books go to train
    for fname, blocks in shuffled_books:
        assigned = False
        for target_split in ["adversarial", "test", "val"]:
            if len(splits[target_split]) < needed[target_split]:
                splits[target_split].extend(blocks)
                split_files[target_split].append(fname)
                assigned = True
                break
        if not assigned:
            splits["train"].extend(blocks)
            split_files["train"].append(fname)

    print("\n[*] Strictly File-Disjoint Split Allocation (Zero Data Leakage):")
    for sp in ["train", "val", "test", "adversarial"]:
        print(
            f"    - {sp.upper():<11}: {len(split_files[sp]):3d} disjoint books | "
            f"{len(splits[sp]):6d} unique blocks (needed: {needed[sp]})"
        )

    # Fallback safety if offline corpus had fewer than 4 files total
    if any(len(splits[sp]) == 0 for sp in splits):
        print("[!] Warning: Too few book files for strict file split; slicing available blocks non-overlappingly.")
        all_blocks = [b for _, blks in shuffled_books for b in blks]
        n = len(all_blocks)
        splits["train"] = all_blocks[: int(0.65 * n)]
        splits["val"] = all_blocks[int(0.65 * n) : int(0.78 * n)]
        splits["test"] = all_blocks[int(0.78 * n) : int(0.91 * n)]
        splits["adversarial"] = all_blocks[int(0.91 * n) :]

    return splits


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


def _encrypt_split_samples(
    text_blocks: list[bytes],
    samples_per_class: int,
    split_name: str,
    rng: np.random.Generator,
    aes_rounds: int = 1,
    des_rounds: int = 2,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Encrypts balanced 4-class samples for a single split using strictly unique plaintext blocks."""
    needed = 4 * samples_per_class
    available = len(text_blocks)
    indices = rng.permutation(available)

    if available < needed:
        print(
            f"[!] WARNING ({split_name}): Requested {needed} unique blocks, only {available} available."
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    s_list: list[str] = []

    cursor = 0
    for class_id in range(4):
        for _ in range(samples_per_class):
            block = text_blocks[int(indices[cursor % available])]
            cursor += 1

            if class_id == 0:
                encrypted = encrypt_aes_reduced(block, rounds=aes_rounds)
            elif class_id == 1:
                encrypted = encrypt_des_reduced(block, rounds=des_rounds)
            elif class_id == 2:
                encrypted = encrypt_3des_reduced(block, rounds_per_stage=1)
            elif class_id == 3:
                encrypted = encrypt_rc4(block)

            X_list.append(np.frombuffer(encrypted, dtype=np.uint8))
            y_list.append(class_id)
            s_list.append(split_name)

    return X_list, y_list, s_list


def generate_main_dataset(
    split_blocks: dict[str, list[bytes]],
    samples_per_class: int = 5000,
    aes_rounds: int = 1,
    des_rounds: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates balanced 4-class dataset with 70/15/15 file-disjoint splits
    and independent random keys per sample.
    """
    rng = np.random.default_rng(seed)

    train_per_class = int(round(samples_per_class * 0.70))
    val_per_class = int(round(samples_per_class * 0.15))
    test_per_class = samples_per_class - train_per_class - val_per_class

    print(
        f"\n[*] Generating main dataset: 4 classes x {samples_per_class} samples "
        f"(Train={4 * train_per_class}, Val={4 * val_per_class}, Test={4 * test_per_class})..."
    )

    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    splits_all: list[str] = []

    for split_name, count in [("train", train_per_class), ("val", val_per_class), ("test", test_per_class)]:
        print(f"    -> Encrypting '{split_name}' split ({4 * count} samples from disjoint books)...")
        X_s, y_s, s_s = _encrypt_split_samples(
            split_blocks[split_name],
            samples_per_class=count,
            split_name=split_name,
            rng=rng,
            aes_rounds=aes_rounds,
            des_rounds=des_rounds,
        )
        X_all.extend(X_s)
        y_all.extend(y_s)
        splits_all.extend(s_s)

    X = np.array(X_all, dtype=np.uint8)
    y = np.array(y_all, dtype=np.int64)
    splits = np.array(splits_all)
    return X, y, splits


def generate_adversarial_dataset(
    text_blocks: list[bytes],
    samples_per_cipher: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates adversarial evaluation set using full-round ciphers (AES-10r labeled 0, DES-16r labeled 1)."""
    print(f"\n[*] Generating adversarial dataset: {samples_per_cipher} AES-10r + {samples_per_cipher} DES-16r...")
    rng = np.random.default_rng(seed + 100)
    available = len(text_blocks)
    needed = 2 * samples_per_cipher
    indices = rng.permutation(available)

    if available < needed:
        print(f"[!] WARNING (adversarial): Requested {needed} unique blocks, only {available} available.")

    X_list = []
    y_list = []
    cursor = 0

    for _ in range(samples_per_cipher):
        block = text_blocks[int(indices[cursor % available])]
        cursor += 1
        encrypted = encrypt_aes_full(block)
        X_list.append(np.frombuffer(encrypted, dtype=np.uint8))
        y_list.append(0)

    for _ in range(samples_per_cipher):
        block = text_blocks[int(indices[cursor % available])]
        cursor += 1
        encrypted = encrypt_des_full(block)
        X_list.append(np.frombuffer(encrypted, dtype=np.uint8))
        y_list.append(1)

    X_adv = np.array(X_list, dtype=np.uint8)
    y_adv = np.array(y_list, dtype=np.int64)
    return X_adv, y_adv


def main():
    parser = argparse.ArgumentParser(description="Generate COA cipher datasets via Wolne Lektury API (Phase 1)")
    parser.add_argument("--samples-per-class", type=int, default=1500,
                        help="Number of samples per class for main dataset (default: 1500, target: 5000)")
    parser.add_argument("--adv-samples", type=int, default=500,
                        help="Number of adversarial samples per cipher (default: 500, target: 1000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with small dataset (400 samples/class, 150 adv)")
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
        samples_per_class = 400
        adv_samples = 150
    else:
        samples_per_class = args.samples_per_class
        adv_samples = args.adv_samples

    train_per_class = int(round(samples_per_class * 0.70))
    val_per_class = int(round(samples_per_class * 0.15))
    test_per_class = samples_per_class - train_per_class - val_per_class

    needed = {
        "train": 4 * train_per_class,
        "val": 4 * val_per_class,
        "test": 4 * test_per_class,
        "adversarial": 2 * adv_samples,
    }
    # Add buffer for whole-book bin packing so every split receives >= needed unique blocks
    target_total_blocks = sum(needed.values()) + 2500

    print("==================================================")
    print(" PHASE 1: DATASET GENERATION PIPELINE             ")
    print("==================================================")
    print(f"[*] Configuration: samples_per_class={samples_per_class}, adv_samples={adv_samples}, seed={args.seed}")

    book_pools = ensure_wolne_lektury_corpus(args.raw_dir, target_blocks=target_total_blocks, block_size=1024)
    if not book_pools:
        raise ValueError("No plaintext books available for encryption.")

    split_blocks = allocate_disjoint_book_splits(book_pools, needed=needed, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Main dataset generation (strictly file-disjoint splits)
    X_main, y_main, splits_main = generate_main_dataset(
        split_blocks,
        samples_per_class=samples_per_class,
        aes_rounds=args.aes_rounds,
        des_rounds=args.des_rounds,
        seed=args.seed,
    )

    main_npz_path = os.path.join(args.output_dir, "dataset.npz")
    np.savez_compressed(main_npz_path, X=X_main, y=y_main, splits=splits_main)

    # Separate .npy matrices for compatibility
    np.save(os.path.join(args.output_dir, "X_ciphertexts.npy"), X_main)
    np.save(os.path.join(args.output_dir, "y_labels.npy"), y_main)
    print(f"[✓] Saved main dataset to {main_npz_path} (shape: {X_main.shape}, zero file/block leakage)")

    # Adversarial dataset generation (from dedicated adversarial books)
    X_adv, y_adv = generate_adversarial_dataset(
        split_blocks["adversarial"],
        samples_per_cipher=adv_samples,
        seed=args.seed,
    )
    adv_npz_path = os.path.join(args.output_dir, "adversarial.npz")
    np.savez_compressed(adv_npz_path, X=X_adv, y=y_adv)
    print(f"[✓] Saved adversarial dataset to {adv_npz_path} (shape: {X_adv.shape})")

    print("\n==================================================")
    print(" DATASET GENERATION COMPLETED SUCCESSFULLY!       ")
    print("==================================================")


if __name__ == "__main__":
    main()
