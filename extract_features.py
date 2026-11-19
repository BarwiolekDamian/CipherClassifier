"""
extract_features.py
Phase 2: Statistical feature extraction and Exploratory Data Analysis (EDA) pipeline
for 1024-byte ciphertext samples.

Computes 5 mandatory MVP features and cryptographically motivated optional features (Section 5.3):
Mandatory (5):
- entropy        : Shannon entropy of 256-value byte distribution
- monobit_stat   : NIST SP 800-22 Frequency (Monobit) test statistic
- ic             : Index of Coincidence across bytes
- autocorr_1     : Byte sequence autocorrelation at lag-1
- autocorr_2     : Byte sequence autocorrelation at lag-2

Optional (30):
- chi2_stat          : Chi-Square goodness-of-fit against uniform byte distribution
- runs_stat          : NIST SP 800-22 Runs test statistic
- autocorr_8         : Autocorrelation lag-8 (64-bit DES/3DES block boundary)
- autocorr_16        : Autocorrelation lag-16 (128-bit AES block boundary)
- iqr                : Interquartile range (Q75 - Q25) of byte values
- max_byte_freq      : Maximum byte frequency count in the 1024-byte sample
- block8_mean_0..7   : Signed byte-mean deviations (centered at 127.5) per offset in 8-byte ECB blocks
- block8_msb_0..7    : Signed MSB-bit deviations (centered at 0.5) per offset in 8-byte ECB blocks
- block64_corr_0..7  : Top-8 strongest SIGNED pairwise bit correlations (preserving +/- sign) in 64-bit ECB blocks
                       (Symmetric around 0 across random keys -> captures non-linear 2nd-order Feistel coupling)

Also generates Phase 2 EDA visualizations in figures/:
- figures/eda_boxplots.png
- figures/eda_correlation_matrix.png
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["AES-red", "DES-red", "3DES-red", "RC4"]

FEATURE_NAMES = (
    [
        "entropy",
        "monobit_stat",
        "ic",
        "autocorr_1",
        "autocorr_2",
        "chi2_stat",
        "runs_stat",
        "autocorr_8",
        "autocorr_16",
        "iqr",
        "max_byte_freq",
    ]
    + [f"block8_mean_{i}" for i in range(8)]
    + [f"block8_msb_{i}" for i in range(8)]
    + [f"block64_corr_{i}" for i in range(8)]
)

_TRIU_64 = np.triu(np.ones((64, 64), dtype=bool), k=1)


def extract_features_from_sample(byte_array: bytes) -> np.ndarray:
    """
    Extracts statistical features from a single 1024-byte sample.
    Returns a 1D numpy array of shape (35,).
    """
    arr = np.frombuffer(byte_array, dtype=np.uint8)
    n = len(arr)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / n

    # 1. Shannon entropy across 256 byte values
    entropy = float(-np.sum(probs * np.log2(probs)))

    # 2. NIST SP 800-22 Monobit test
    bit_arr = np.unpackbits(arr)
    n_bits = len(bit_arr)
    s_n = np.abs(np.sum(2 * bit_arr.astype(int) - 1))
    monobit_stat = float(s_n / np.sqrt(n_bits))

    # 3. Index of Coincidence (IC)
    ic = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))

    # Variance computation with zero-division safeguard
    arr_f = arr.astype(np.float64) - np.mean(arr)
    var = np.sum(arr_f ** 2)
    safe_var = var if var > 1e-9 else 1.0

    # 4 & 5. Autocorrelation lag-1 and lag-2
    autocorr_1 = float(np.sum(arr_f[:-1] * arr_f[1:]) / safe_var) if var > 1e-9 else 0.0
    autocorr_2 = float(np.sum(arr_f[:-2] * arr_f[2:]) / safe_var) if var > 1e-9 else 0.0

    # 6. Chi-Square statistic against uniform byte distribution
    expected = n / 256.0
    chi2_stat, _ = chisquare(counts, f_exp=np.full(256, expected))
    chi2_stat = float(chi2_stat)

    # 7. NIST SP 800-22 Runs test for bit sequences
    pi = np.mean(bit_arr)
    if abs(pi - 0.5) >= (2.0 / np.sqrt(n_bits)):
        runs_stat = 0.0
    else:
        v_n = np.sum(bit_arr[:-1] != bit_arr[1:]) + 1
        numerator = abs(v_n - 2.0 * n_bits * pi * (1.0 - pi))
        denominator = 2.0 * np.sqrt(2.0 * n_bits) * pi * (1.0 - pi)
        runs_stat = float(numerator / (denominator + 1e-9))

    # 8 & 9. Autocorrelation lag-8 (DES/3DES 64-bit block) and lag-16 (AES 128-bit block)
    autocorr_8 = float(np.sum(arr_f[:-8] * arr_f[8:]) / safe_var) if var > 1e-9 else 0.0
    autocorr_16 = float(np.sum(arr_f[:-16] * arr_f[16:]) / safe_var) if var > 1e-9 else 0.0

    # 10 & 11. Interquartile range (IQR) and max byte frequency
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    max_byte_freq = float(np.max(counts))

    # 12-19. Signed per-offset byte means in 8-byte ECB blocks (centered at 127.5)
    blocks_8 = arr.reshape(-1, 8).astype(np.float64)
    col_means = (np.mean(blocks_8, axis=0) - 127.5).tolist()

    # 20-27. Signed per-offset MSB bit bias in 8-byte ECB blocks (centered at 0.5)
    msb_8 = ((arr.reshape(-1, 8) >> 7) & 1).astype(np.float64)
    msb_bias = (np.mean(msb_8, axis=0) - 0.5).tolist()

    # 28-35. Top-8 strongest SIGNED pairwise bit correlations across 64-bit ECB blocks (preserving +/- sign)
    bits_64 = bit_arr.reshape(-1, 64).astype(np.float64)
    bc = bits_64 - bits_64.mean(axis=0, keepdims=True)
    bn = bc / (bc.std(axis=0, keepdims=True) + 1e-9)
    corr_u = ((bn.T @ bn) / len(bits_64))[_TRIU_64]
    top_idx = np.argpartition(np.abs(corr_u), -8)[-8:]
    top_idx = top_idx[np.argsort(np.abs(corr_u[top_idx]))[::-1]]
    top_signed_corrs = corr_u[top_idx].tolist()

    return np.array(
        [
            entropy,
            monobit_stat,
            ic,
            autocorr_1,
            autocorr_2,
            chi2_stat,
            runs_stat,
            autocorr_8,
            autocorr_16,
            iqr,
            max_byte_freq,
        ]
        + col_means
        + msb_bias
        + top_signed_corrs,
        dtype=np.float64,
    )


def extract_features_batch(samples: list[bytes]) -> np.ndarray:
    """Extracts features for a batch of byte samples."""
    return np.array([extract_features_from_sample(s) for s in samples], dtype=np.float64)


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "figures"):
    """
    Generates Phase 2 Exploratory Data Analysis (EDA) artifacts:
    - Class balance check & NaN sanity report
    - Boxplots of key statistical features per cipher family
    - Feature correlation matrix heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) ---")

    # Class balance and quality check
    counts = df["label"].value_counts().sort_index()
    print("[*] Class balance check:")
    for cls_id, cnt in counts.items():
        name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"Class_{cls_id}"
        print(f"    - {name:<10} (label={cls_id}): {cnt} samples")

    nan_count = int(df[FEATURE_NAMES].isna().sum().sum())
    print(f"[*] Missing / NaN values across all features: {nan_count}")

    # 1. Boxplots per feature and cipher class (6 key features)
    key_features = [
        ("entropy", "Shannon Entropy [bits]"),
        ("chi2_stat", "Chi-Square Statistic"),
        ("ic", "Index of Coincidence (IC)"),
        ("autocorr_8", "Autocorrelation Lag-8"),
        ("block8_mean_0", "ECB Block-8 Offset-0 Mean Dev."),
        ("block8_msb_0", "ECB Block-8 Offset-0 MSB Bias"),
    ]
    present_classes = sorted(df["label"].unique())
    present_names = [CLASS_NAMES[int(c)] for c in present_classes]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (feat, title) in zip(axes.flat, key_features):
        data_per_class = [df.loc[df["label"] == c, feat].values for c in present_classes]
        bp = ax.boxplot(data_per_class, tick_labels=present_names, patch_artist=True, showfliers=False)
        colors = ["#3b75af", "#ef8636", "#519e3e", "#c53932"]
        for patch, color in zip(bp["boxes"], colors[: len(present_classes)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("EDA: Statistical Feature Distributions by Cipher Family", fontsize=13, fontweight="bold")
    plt.tight_layout()
    box_path = os.path.join(output_dir, "eda_boxplots.png")
    plt.savefig(box_path, dpi=300)
    plt.close()

    # 2. Correlation matrix heatmap across core statistical features
    corr_cols = (
        FEATURE_NAMES[:11]
        + ["block8_mean_0", "block8_msb_0", "block64_corr_0", "block64_corr_1"]
    )
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_cols, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
    ax.set_title("EDA: Statistical Feature Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "eda_correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()

    print(f"[✓] Saved EDA artifacts to {box_path} and {corr_path}")


def process_dataset_file(input_path: str, output_path: str, run_eda: bool = False):
    """Loads a .npz file, verifies constant 1024-byte sample length, extracts features, and saves CSV."""
    if not os.path.exists(input_path):
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"\n[*] Loading raw samples from {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    raw_samples = data["X"]
    labels = data["y"]
    splits = data["splits"] if "splits" in data.files else None

    # Verify constant sample length (anti-leakage requirement, Section 4)
    lengths = {len(s.tobytes() if hasattr(s, "tobytes") else bytes(s)) for s in raw_samples}
    assert lengths == {1024}, f"Data leakage alert! Variable sample lengths detected: {lengths}"
    print(f"[*] Verified constant sample length: 1024 bytes across all {len(raw_samples)} samples.")

    print(f"[*] Extracting {len(FEATURE_NAMES)} statistical features for {len(raw_samples)} samples...")
    t_start = time.perf_counter()
    feature_matrix = []
    for s in raw_samples:
        byte_data = s.tobytes() if hasattr(s, "tobytes") else bytes(s)
        feature_matrix.append(extract_features_from_sample(byte_data))

    total_time = time.perf_counter() - t_start
    print(f"[*] Extraction completed in {total_time:.2f} seconds ({total_time / len(raw_samples) * 1000:.3f} ms/sample).")

    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    df["label"] = labels
    if splits is not None:
        df["split"] = splits
    df.to_csv(output_path, index=False)
    print(f"[✓] Successfully saved extracted features to {output_path}")

    if run_eda:
        generate_eda_plots(df)

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
    parser = argparse.ArgumentParser(description="Statistical feature extraction & EDA pipeline (Phase 2)")
    parser.add_argument("--input", type=str, default="data/ciphertexts/dataset.npz",
                        help="Path to .npz file containing 'X' (samples) and 'y' (labels)")
    parser.add_argument("--output", type=str, default="data/features.csv",
                        help="Path to output CSV file for main dataset")
    parser.add_argument("--adv-input", type=str, default="data/ciphertexts/adversarial.npz",
                        help="Path to adversarial .npz dataset file")
    parser.add_argument("--adv-output", type=str, default="data/adversarial_features.csv",
                        help="Path to output CSV file for adversarial dataset")
    parser.add_argument("--no-eda", action="store_true",
                        help="Skip EDA plot generation")
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

    # Process main dataset + generate Phase 2 EDA figures
    process_dataset_file(args.input, args.output, run_eda=not args.no_eda)

    # Automatically process adversarial dataset if present
    if os.path.exists(args.adv_input):
        print(f"\n[*] Detected adversarial dataset at {args.adv_input}. Extracting...")
        process_dataset_file(args.adv_input, args.adv_output, run_eda=False)


if __name__ == "__main__":
    main()
