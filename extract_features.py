"""
extract_features.py
Statistical feature extraction pipeline for 1024-byte ciphertext samples.
Computes mandatory MVP features and cryptographically motivated optional features:
- Shannon entropy (byte distribution)
- NIST SP 800-22 Monobit test
- Index of Coincidence (IC)
- Autocorrelation lag-1
- Autocorrelation lag-2
- Chi-Square goodness-of-fit against uniform byte distribution
- Autocorrelation lag-8 (sensitive to 64-bit DES/3DES block boundaries)
- NIST SP 800-22 Runs test (bit run count)
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
    "chi2_stat",
    "autocorr_8",
    "runs_stat"
]


def extract_features_from_sample(byte_array: bytes) -> np.ndarray:
    """
    Extracts statistical features from a single 1024-byte sample.
    Returns a 1D numpy array of shape (8,).
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

    # Autocorrelation lag-1 and lag-2
    autocorr_1 = float(np.sum(arr_f[:-1] * arr_f[1:]) / safe_var) if var > 1e-9 else 0.0
    autocorr_2 = float(np.sum(arr_f[:-2] * arr_f[2:]) / safe_var) if var > 1e-9 else 0.0

    # Chi-Square statistic against uniform byte distribution
    expected = n / 256.0
    chi2_stat, _ = chisquare(counts, f_exp=np.full(256, expected))
    chi2_stat = float(chi2_stat)

    # Autocorrelation lag-8 to detect 64-bit DES block boundaries
    autocorr_8 = float(np.sum(arr_f[:-8] * arr_f[8:]) / safe_var) if var > 1e-9 else 0.0

    # NIST SP 800-22 Runs test for bit sequences
    pi = np.mean(bit_arr)
    if abs(pi - 0.5) >= (2.0 / np.sqrt(n_bits)):
        runs_stat = 0.0
    else:
        v_n = np.sum(bit_arr[:-1] != bit_arr[1:]) + 1
        numerator = abs(v_n - 2.0 * n_bits * pi * (1.0 - pi))
        denominator = 2.0 * np.sqrt(2.0 * n_bits) * pi * (1.0 - pi)
        runs_stat = float(numerator / (denominator + 1e-9))

    return np.array([
        entropy,
        monobit_stat,
        ic,
        autocorr_1,
        autocorr_2,
        chi2_stat,
        autocorr_8,
        runs_stat
    ], dtype=np.float64)


def extract_features_batch(samples: list[bytes]) -> np.ndarray:
    """Extracts features for a batch of byte samples."""
    return np.array([extract_features_from_sample(s) for s in samples], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Statistical feature extraction pipeline (Phase 2)")
    parser.add_argument("--input", type=str, default="data/ciphertexts/dataset.npz",
                        help="Path to .npz file containing 'X' (samples) and 'y' (labels)")
    parser.add_argument("--output", type=str, default="data/features.csv",
                        help="Path to output CSV file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if not args.input.endswith(".npz"):
        print("[!] Please provide a path to a .npz dataset file containing arrays 'X' and 'y'.")
        return

    if not os.path.exists(args.input):
        print(f"[!] Input file not found: {args.input}")
        return

    print(f"[*] Loading raw samples from {args.input}...")
    data = np.load(args.input, allow_pickle=True)
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
    df.to_csv(args.output, index=False)
    print(f"[✓] Successfully saved extracted features to {args.output}")


if __name__ == "__main__":
    import sys

    # If CLI arguments are passed, execute main extraction pipeline
    if len(sys.argv) > 1:
        main()
    else:
        # Run test verification suite
        print("==================================================")
        print(" RUNNING FEATURE EXTRACTION VERIFICATION SUITE    ")
        print("==================================================")

        # Test on repeated byte sample
        sample_uniform = bytes(range(256)) * 4
        print(f"[*] Sample size: {len(sample_uniform)} bytes")

        print("\n--- Extracting features from test sample ---")
        feats = extract_features_from_sample(sample_uniform)
        print(f"  Feature vector length: {len(feats)}")

        for name, val in zip(FEATURE_NAMES, feats):
            print(f"  - {name:<15}: {val:12.6f}")

        assert len(feats) == len(FEATURE_NAMES), "Feature count must match FEATURE_NAMES"
        assert not np.isnan(feats).any(), "Extracted features must not contain NaN"
        assert not np.isinf(feats).any(), "Extracted features must not contain Inf"
        print("\n  Validation status    : [PASSED]")

        print("==================================================")
        print(" ALL TESTS PASSED: Feature extractor is functional!")
        print("==================================================")
