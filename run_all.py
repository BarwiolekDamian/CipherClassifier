"""
run_all.py
End-to-End Orchestrator for Cipher Family Classification Pipeline (Phases 1-5).

Executes the entire research and deployment pipeline in a single command:
  1. Phase 1: Dataset Generation (generate_data.py)
  2. Phase 2: Feature Extraction (extract_features.py)
  3. Phases 3 & 4: Model Training, Evaluation & Adversarial COA Test (train_model.py)
  4. Phase 5: Inference Verification & Demo Prediction (predict.py)
"""

import argparse
import os
import subprocess
import sys
import time


def print_banner(text: str, char: str = "="):
    border = char * 65
    print(f"\n{border}")
    print(f" {text}")
    print(f"{border}\n")


def run_command(cmd: list[str], phase_name: str) -> float:
    """Runs a pipeline step as a subprocess, streaming logs and timing execution."""
    print_banner(f"STARTING {phase_name.upper()}")
    print(f"[*] Command: {' '.join(cmd)}\n")

    t_start = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print_banner(f"ERROR: {phase_name.upper()} FAILED (exit code {e.returncode})", char="!")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(130)

    elapsed = time.perf_counter() - t_start
    print(f"\n[✓] {phase_name} finished successfully in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run complete CipherClassifier pipeline end-to-end (Phases 1-5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", action="store_true",
        help="Quick run with smaller sample sizes (350/class) to verify pipeline fast (~30-60s)"
    )
    group.add_argument(
        "--full", action="store_true",
        help="Full production run with target dataset (5000/class, 1000 adv per specification)"
    )

    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip Phase 1 (dataset generation) if data/ciphertexts/ already exists"
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip Phase 2 (feature extraction) if data/features.csv already exists"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip Phase 3 & 4 (training) and jump straight to Phase 5 inference"
    )
    parser.add_argument(
        "--demo-predict", action="store_true", default=True,
        help="Run Phase 5 sample prediction at the end"
    )

    args = parser.parse_args()

    python_bin = sys.executable
    total_start = time.perf_counter()
    step_times = {}

    print_banner("CIPHER FAMILY CLASSIFIER - END-TO-END PIPELINE")
    print(f"[*] Python Interpreter: {python_bin}")
    print(f"[*] Mode: {'FULL (5000 samples/class)' if args.full else ('QUICK (350 samples/class)' if args.quick else 'DEFAULT (1500 samples/class)')}")

    # Ensure required directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/ciphertexts", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 1: DATASET GENERATION
    # -------------------------------------------------------------
    dataset_npz = "data/ciphertexts/dataset.npz"
    if args.skip_data and os.path.exists(dataset_npz):
        print(f"\n[*] Skipping Phase 1: Found existing dataset at {dataset_npz}")
    else:
        gen_cmd = [python_bin, "generate_data.py"]
        if args.full:
            gen_cmd.append("--full")
        elif args.quick:
            gen_cmd.append("--quick")

        step_times["Phase 1 (Data Gen)"] = run_command(gen_cmd, "Phase 1: Dataset Generation")

    # -------------------------------------------------------------
    # PHASE 2: FEATURE EXTRACTION
    # -------------------------------------------------------------
    features_csv = "data/features.csv"
    if args.skip_features and os.path.exists(features_csv):
        print(f"\n[*] Skipping Phase 2: Found existing features at {features_csv}")
    else:
        feat_cmd = [
            python_bin, "extract_features.py",
            "--input", "data/ciphertexts/dataset.npz",
            "--output", "data/features.csv",
            "--adv-input", "data/ciphertexts/adversarial.npz",
            "--adv-output", "data/adversarial_features.csv"
        ]
        step_times["Phase 2 (Feature Extraction)"] = run_command(feat_cmd, "Phase 2: Feature Extraction")

    # -------------------------------------------------------------
    # PHASES 3 & 4: MODEL TRAINING & ADVERSARIAL TEST
    # -------------------------------------------------------------
    if args.skip_train:
        print("\n[*] Skipping Phases 3 & 4: Training skipped by user flag.")
    else:
        train_cmd = [
            python_bin, "train_model.py",
            "--features", "data/features.csv",
            "--adversarial", "data/adversarial_features.csv"
        ]
        step_times["Phases 3 & 4 (Training & Eval)"] = run_command(train_cmd, "Phases 3 & 4: Model Training & Evaluation")

    # -------------------------------------------------------------
    # PHASE 5: INFERENCE VERIFICATION
    # -------------------------------------------------------------
    if args.demo_predict:
        pred_cmd = [
            python_bin, "predict.py",
            "--model", "models/rf_4class_final.joblib"
        ]
        step_times["Phase 5 (Predict Verification)"] = run_command(pred_cmd, "Phase 5: Inference CLI Verification")

    # -------------------------------------------------------------
    # PIPELINE SUMMARY
    # -------------------------------------------------------------
    total_time = time.perf_counter() - total_start
    print_banner("PIPELINE EXECUTION SUMMARY")
    for step, duration in step_times.items():
        print(f"  - {step:<32}: {duration:8.2f} s")
    print("-" * 65)
    print(f"  TOTAL EXECUTION TIME         : {total_time:8.2f} s ({total_time / 60:.2f} min)")
    print("\n[✓] Generated Artifacts:")
    print("  - Models  : models/rf_4class_final.joblib, models/rf_binary_mvp.joblib, models/baseline_*.joblib")
    print("  - Figures : figures/confusion_matrix.png, figures/roc_curves.png, figures/feature_importance.png")
    print("  - Data    : data/features.csv, data/adversarial_features.csv")
    print_banner("ALL PHASES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

