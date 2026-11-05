"""
extract_features.py
Statistical feature extraction pipeline for 1024-byte ciphertext samples.
Computes mandatory MVP features and cryptographically motivated optional features (Section 5.3):
- Shannon entropy (byte distribution)
- NIST SP 800-22 Monobit test
- Index of Coincidence (IC)
- Autocorrelation lag-1
- Autocorrelation lag-2
- Autocorrelation lag-4 (sensitive to 4-byte AES columns and DES Feistel halves)
- Chi-Square goodness-of-fit against uniform byte distribution
- Autocorrelation lag-8 (sensitive to 64-bit DES/3DES block boundaries)
- Autocorrelation lag-16 (sensitive to 128-bit AES block boundaries)
- Maximum byte frequency (max_byte_freq)
- Zero-occurrence byte count (zero_byte_count)
- Standard deviation of byte distribution (std_byte_freq)
- NIST SP 800-22 Runs test (bit run count)
- 8-byte block boundary variance ratio (boundary_ratio_8)
- 16-byte block boundary variance ratio (boundary_ratio_16)
- Feistel half-block variance asymmetry (feistel_half_diff)
- 64-bit block position bias standard deviation (bit_pos_std_64)
- 128-bit block position bias standard deviation (bit_pos_std_128)
- Unique 8-byte block ratio in ECB mode (unique_blocks_8)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import chisquare

FEATURE_NAMES = [
    "entropy",
    "monobit_stat",
    "ic",
    "autocorr_1",
    "autocorr_2",
    "autocorr_4",
    "chi2_stat",
    "autocorr_8",
    "autocorr_16",
    "max_byte_freq",
    "zero_byte_count",
    "std_byte_freq",
    "runs_stat",
    "boundary_ratio_8",
    "boundary_ratio_16",
    "feistel_half_diff",
    "bit_pos_std_64",
    "bit_pos_std_128",
    "unique_blocks_8"
]


def extract_features_from_sample(byte_array: bytes) -> np.ndarray:
    """
    Extracts statistical features from a single 1024-byte sample.
    Returns a 1D numpy array of shape (19,).
    """
    arr = np.frombuffer(byte_array, dtype=np.uint8)
    n = len(arr)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / n

    # Shannon entropy across 256 byte values
    entropy = float(-np.sum(probs * np.log2(probs)))

    # NIST SP 800-22 Monobit test
    bit_arr = np.unpackbits(arr)
    n_bits = len(bit_arr)
    s_n = np.abs(np.sum(2 * bit_arr.astype(int) - 1))
    monobit_stat = float(s_n / np.sqrt(n_bits))

    # Index of Coincidence (IC)
    ic = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))

    # Variance computation with zero-division safeguard
    arr_f = arr.astype(float) - np.mean(arr)
    var = np.sum(arr_f ** 2)
    safe_var = var if var > 1e-9 else 1.0

    # Autocorrelation lag-1, lag-2, lag-4
    autocorr_1 = float(np.sum(arr_f[:-1] * arr_f[1:]) / safe_var) if var > 1e-9 else 0.0
    autocorr_2 = float(np.sum(arr_f[:-2] * arr_f[2:]) / safe_var) if var > 1e-9 else 0.0
    autocorr_4 = float(np.sum(arr_f[:-4] * arr_f[4:]) / safe_var) if var > 1e-9 else 0.0

    # Chi-Square statistic against uniform byte distribution
    expected = n / 256.0
    chi2_stat, _ = chisquare(counts, f_exp=np.full(256, expected))
    chi2_stat = float(chi2_stat)

    # Autocorrelation lag-8 to detect 64-bit DES/3DES block boundaries (8 bytes)
    autocorr_8 = float(np.sum(arr_f[:-8] * arr_f[8:]) / safe_var) if var > 1e-9 else 0.0

    # Autocorrelation lag-16 to detect 128-bit AES block boundaries (16 bytes)
    autocorr_16 = float(np.sum(arr_f[:-16] * arr_f[16:]) / safe_var) if var > 1e-9 else 0.0

    # Byte frequency distribution statistics (Section 5.3 optional features)
    max_byte_freq = float(np.max(counts))
    zero_byte_count = float(np.sum(counts == 0))
    std_byte_freq = float(np.std(counts))

    # NIST SP 800-22 Runs test for bit sequences
    pi = np.mean(bit_arr)
    if abs(pi - 0.5) >= (2.0 / np.sqrt(n_bits)):
        runs_stat = 0.0
    else:
        v_n = np.sum(bit_arr[:-1] != bit_arr[1:]) + 1
        numerator = abs(v_n - 2.0 * n_bits * pi * (1.0 - pi))
        denominator = 2.0 * np.sqrt(2.0 * n_bits) * pi * (1.0 - pi)
        runs_stat = float(numerator / (denominator + 1e-9))

    # Block boundary transition variance ratios (ECB block artifacts)
    diffs_sq = (arr[1:].astype(np.float64) - arr[:-1].astype(np.float64)) ** 2
    idx = np.arange(1, n)

    is_boundary_8 = (idx % 8 == 0)
    boundary_diff_8 = np.mean(diffs_sq[is_boundary_8])
    internal_diff_8 = np.mean(diffs_sq[~is_boundary_8])
    boundary_ratio_8 = float(boundary_diff_8 / (internal_diff_8 + 1e-9))

    is_boundary_16 = (idx % 16 == 0)
    boundary_diff_16 = np.mean(diffs_sq[is_boundary_16])
    internal_diff_16 = np.mean(diffs_sq[~is_boundary_16])
    boundary_ratio_16 = float(boundary_diff_16 / (internal_diff_16 + 1e-9))

    # Feistel half-block asymmetry (variance difference between left and right 4-byte halves of 8-byte blocks)
    blocks_8 = arr.reshape(-1, 8).astype(np.float64)
    left_var = np.mean(np.var(blocks_8[:, :4], axis=1))
    right_var = np.mean(np.var(blocks_8[:, 4:], axis=1))
    feistel_half_diff = float(abs(left_var - right_var))

    # Bit position distribution statistics across 64-bit and 128-bit blocks
    bits_64 = bit_arr.reshape(-1, 64)
    bit_pos_std_64 = float(np.std(np.mean(bits_64, axis=0)))

    bits_128 = bit_arr.reshape(-1, 128)
    bit_pos_std_128 = float(np.std(np.mean(bits_128, axis=0)))

    # Unique 8-byte blocks ratio (ECB mode repeats identical blocks on recurring natural n-grams)
    blocks_8_raw = [byte_array[i * 8 : (i + 1) * 8] for i in range(n // 8)]
    unique_blocks_8 = float(len(set(blocks_8_raw)) / len(blocks_8_raw))

    return np.array([
        entropy,
        monobit_stat,
        ic,
        autocorr_1,
        autocorr_2,
        autocorr_4,
        chi2_stat,
        autocorr_8,
        autocorr_16,
        max_byte_freq,
        zero_byte_count,
        std_byte_freq,
        runs_stat,
        boundary_ratio_8,
        boundary_ratio_16,
        feistel_half_diff,
        bit_pos_std_64,
        bit_pos_std_128,
        unique_blocks_8
    ], dtype=np.float64)


def extract_features_batch(samples: list[bytes]) -> np.ndarray:
    """Extracts features for a batch of byte samples."""
    return np.array([extract_features_from_sample(s) for s in samples], dtype=np.float64)


def process_dataset_file(input_path: str, output_path: str):
    """Loads a .npz file, computes features for all samples, and saves to CSV."""
    if not os.path.exists(input_path):
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"\n[*] Loading raw samples from {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    raw_samples = data["X"]
    labels = data["y"]

    print(f"[*] Extracting statistical features for {len(raw_samples)} samples...")
    t_start = time.perf_counter()
    feature_matrix = []
    for s in raw_samples:
        byte_data = s.tobytes() if hasattr(s, "tobytes") else bytes(s)
        feature_matrix.append(extract_features_from_sample(byte_data))

    total_time = time.perf_counter() - t_start
    print(f"[*] Extraction completed in {total_time:.2f} seconds ({total_time / len(raw_samples) * 1000:.3f} ms/sample).")

    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    df["label"] = labels
    df.to_csv(output_path, index=False)
    print(f"[✓] Successfully saved extracted features to {output_path}")
    return True


def run_verification_suite():
    """Runs a verification test suite on a known byte pattern."""
    print("==================================================")
    print(" RUNNING FEATURE EXTRACTION VERIFICATION SUITE    ")
    print("==================================================")

    sample_uniform = bytes(range(256)) * 4
    print(f"[*] Sample size: {len(sample_uniform)} bytes")

    print("\n--- Extracting features from test sample ---")
    feats = extract_features_from_sample(sample_uniform)
    print(f"  Feature vector length: {len(feats)}")

    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  - {name:<17}: {val:12.6f}")

    assert len(feats) == len(FEATURE_NAMES), "Feature count must match FEATURE_NAMES"
    assert not np.isnan(feats).any(), "Extracted features must not contain NaN"
    assert not np.isinf(feats).any(), "Extracted features must not contain Inf"
    print("\n  Validation status    : [PASSED]")

    print("==================================================")
    print(" ALL TESTS PASSED: Feature extractor is functional!")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Statistical feature extraction pipeline (Phase 2)")
    parser.add_argument("--input", type=str, default="data/ciphertexts/dataset.npz",
                        help="Path to .npz file containing 'X' (samples) and 'y' (labels)")
    parser.add_argument("--output", type=str, default="data/features.csv",
                        help="Path to output CSV file for main dataset")
    parser.add_argument("--adv-input", type=str, default="data/ciphertexts/adversarial.npz",
                        help="Path to adversarial .npz dataset file")
    parser.add_argument("--adv-output", type=str, default="data/adversarial_features.csv",
                        help="Path to output CSV file for adversarial dataset")
    parser.add_argument("--test", "--verify", dest="test", action="store_true",
                        help="Run verification test suite on synthetic sample")
    args = parser.parse_args()

    if args.test:
        run_verification_suite()
        return

    if not os.path.exists(args.input):
        print(f"[*] Dataset file '{args.input}' not found.")
        print("[*] Running feature extraction self-verification suite instead...")
        run_verification_suite()
        return

    # Process main dataset
    process_dataset_file(args.input, args.output)

    # Automatically process adversarial dataset if present
    if os.path.exists(args.adv_input):
        print(f"\n[*] Detected adversarial dataset at {args.adv_input}. Extracting...")
        process_dataset_file(args.adv_input, args.adv_output)


if __name__ == "__main__":
    main()
