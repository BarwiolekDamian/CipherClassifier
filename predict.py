"""
predict.py
Phase 5: CLI inference tool for Ciphertext-Only Attack (COA) cipher family classification.

MLOps deployment and monitoring capabilities:
- Ingests trained serialized model (.joblib pipeline containing scaler and classifier)
- Supports single sample inspection (--file) and directory batch processing (--dir)
- Computes statistical feature vectors on the fly via extract_features module
- Measures and reports precise inference latency in milliseconds per sample
- Flags uncertain predictions (max probability < 0.50) as security alerts for L2 SOC analysts
"""

import argparse
import os
import sys
import time
import joblib
import numpy as np

from extract_features import extract_features_from_sample

CLASS_NAMES = {
    0: "AES-red",
    1: "DES-red",
    2: "3DES-red",
    3: "RC4"
}


def predict_sample(model, byte_data: bytes) -> dict:
    """Extracts features and performs inference on a single 1024-byte sample."""
    if len(byte_data) != 1024:
        raise ValueError(f"Expected exact 1024-byte sample, received: {len(byte_data)} bytes")

    t_start = time.perf_counter()
    features = extract_features_from_sample(byte_data).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    inference_time_ms = (time.perf_counter() - t_start) * 1000.0

    pred_class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_class_idx])
    is_uncertain = confidence < 0.50

    return {
        "class_idx": pred_class_idx,
        "class_name": CLASS_NAMES.get(pred_class_idx, f"Class_{pred_class_idx}"),
        "confidence": confidence,
        "probabilities": probabilities,
        "is_uncertain": is_uncertain,
        "inference_time_ms": inference_time_ms
    }


def process_single_file(model, file_path: str):
    """Processes a single 1024-byte file and prints a formatted SOC report."""
    if not os.path.exists(file_path):
        print(f"[!] File does not exist: {file_path}")
        return

    with open(file_path, "rb") as f:
        data = f.read(1024)

    if len(data) < 1024:
        print(f"[!] File {file_path} is smaller than 1024 bytes ({len(data)} bytes).")
        return

    res = predict_sample(model, data)

    print("\n" + "=" * 55)
    print(f" SOC COA INFERENCE REPORT: {os.path.basename(file_path)} ")
    print("=" * 55)
    print(f"Predicted Family   : {res['class_name']}")
    print(f"Confidence (max p) : {res['confidence'] * 100:.2f}%")
    print(f"Inference Latency  : {res['inference_time_ms']:.2f} ms")

    if res["is_uncertain"]:
        print("\n[!] STATUS: [ALERT: LOW CONFIDENCE (< 50%) -> ESCALATE TO L2 SOC ANALYST]")
    else:
        print("\n[✓] STATUS: [HIGH CONFIDENCE PREDICTION]")

    print("\nClass Probability Distribution:")
    for idx, prob in enumerate(res["probabilities"]):
        bar = "█" * int(prob * 25)
        print(f"  - {CLASS_NAMES.get(idx, f'Class_{idx}'):<10}: {prob * 100:5.2f}% | {bar}")
    print("=" * 55 + "\n")


def process_directory(model, dir_path: str):
    """Processes all 1024-byte samples in a directory and outputs a batch summary."""
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not files:
        print(f"[!] No files found in directory: {dir_path}")
        return

    print(f"\n[*] Processing {len(files)} samples from {dir_path}...\n")
    print(f"{'Filename':<25} | {'Prediction':<10} | {'Confidence':<10} | {'Latency':<9} | {'Status'}")
    print("-" * 75)

    uncertain_count = 0
    total_time = 0.0
    processed_count = 0

    for fpath in sorted(files):
        try:
            with open(fpath, "rb") as f:
                data = f.read(1024)
            if len(data) != 1024:
                continue

            res = predict_sample(model, data)
            total_time += res["inference_time_ms"]
            processed_count += 1
            status_str = "[FLAGGED: L2]" if res["is_uncertain"] else "[OK]"
            if res["is_uncertain"]:
                uncertain_count += 1

            fname = os.path.basename(fpath)
            if len(fname) > 23:
                fname = fname[:20] + "..."
            print(f"{fname:<25} | {res['class_name']:<10} | {res['confidence'] * 100:6.2f}%    | {res['inference_time_ms']:5.2f} ms  | {status_str}")
        except Exception as e:
            print(f"{os.path.basename(fpath):<25} | ERROR: {e}")

    avg_time = total_time / max(processed_count, 1)
    print("-" * 75)
    print(f"Summary: Successfully processed {processed_count} samples. Average latency: {avg_time:.2f} ms/sample.")
    print(f"Samples flagged for low confidence (< 50%): {uncertain_count} / {processed_count}")


def run_verification_suite(model_path="models/rf_4class_final.joblib"):
    """Runs automated verification of inference pipeline and monitoring capabilities."""
    print("==================================================")
    print(" RUNNING PREDICTION CLI VERIFICATION SUITE        ")
    print("==================================================")

    if not os.path.exists(model_path):
        print(f"[!] Saved model not found at '{model_path}'. Run train_model.py first.")
        return

    print(f"[*] Loading model pipeline from {model_path}...")
    model = joblib.load(model_path)

    test_sample = bytes(range(256)) * 4
    print(f"[*] Testing inference on sample ({len(test_sample)} bytes)...")

    res = predict_sample(model, test_sample)
    print(f"  Predicted family  : {res['class_name']}")
    print(f"  Confidence        : {res['confidence'] * 100:.2f}%")
    print(f"  Inference latency : {res['inference_time_ms']:.2f} ms")
    print(f"  Uncertainty flag  : {res['is_uncertain']} (flagged: {'YES' if res['is_uncertain'] else 'NO'})")

    assert res["class_idx"] in CLASS_NAMES, "Predicted class index must be valid"
    assert res["inference_time_ms"] > 0, "Inference latency must be positive"
    assert len(res["probabilities"]) == len(CLASS_NAMES), "Probabilities must cover all classes"

    print("\n  Validation status : [PASSED]")
    print("==================================================")
    print(" ALL TESTS PASSED: Prediction CLI is operational! ")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Cipher Family COA Inference Tool (Phase 5)")
    parser.add_argument("--file", type=str, help="Path to a single 1024-byte sample file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing 1024-byte sample files")
    parser.add_argument("--model", type=str, default="models/rf_4class_final.joblib",
                        help="Path to serialized model pipeline (.joblib)")
    args = parser.parse_args()

    # If neither file nor dir is specified, run self-verification suite
    if not args.file and not args.dir:
        run_verification_suite(args.model)
        return

    if not os.path.exists(args.model):
        print(f"[!] Model not found at '{args.model}'. Please train the model first via train_model.py.")
        sys.exit(1)

    model = joblib.load(args.model)

    if args.file:
        process_single_file(model, args.file)
    elif args.dir:
        process_directory(model, args.dir)


if __name__ == "__main__":
    main()
