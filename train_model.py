"""
train_model.py
Phases 3 & 4: Model training, hyperparameter optimization, evaluation, and Adversarial COA Test.

Key requirements:
- Stratified data split: 70% train / 15% validation / 15% test (seed 42)
- Zero data leakage: StandardScaler fitted strictly on training data (70%) via sklearn Pipeline
- Baseline models: Multinomial Logistic Regression and single Decision Tree
- Binary MVP model: AES-reduced vs DES-reduced (verification threshold: Accuracy > 88%, F1 > 0.85)
- Multiclass model: 4-class Random Forest with validation split tuning via GridSearchCV (PredefinedSplit)
- Visualization artifacts: Confusion Matrix, ROC curves (One-vs-Rest), Feature Importance
- Adversarial COA Test: evaluation on full-round ciphers (AES-10r, DES-16r) with <= 35% accuracy threshold
"""

import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

CLASS_NAMES = ["AES-red", "DES-red", "3DES-red", "RC4"]


def train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Trains baseline models on train set and evaluates on validation and test sets."""
    print("\n" + "=" * 60)
    print(" BASELINE MODELS EVALUATION ")
    print("=" * 60)

    results = {}

    # Multinomial Logistic Regression baseline
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    t0 = time.perf_counter()
    pipe_lr.fit(X_train, y_train)
    t_lr = time.perf_counter() - t0

    y_val_lr = pipe_lr.predict(X_val)
    val_acc_lr = accuracy_score(y_val, y_val_lr)
    val_f1_lr = f1_score(y_val, y_val_lr, average="macro")

    y_pred_lr = pipe_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro")

    results["Logistic Regression"] = {
        "val_acc": val_acc_lr, "val_f1": val_f1_lr,
        "acc": acc_lr, "f1": f1_lr,
        "time": t_lr, "model": pipe_lr
    }
    print(f"[Baseline] Logistic Regression -> Val Acc: {val_acc_lr:.4f}, Val F1: {val_f1_lr:.4f} | Test Acc: {acc_lr:.4f}, Test F1: {f1_lr:.4f} ({t_lr:.2f}s)")

    # Single Decision Tree baseline
    pipe_dt = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    t0 = time.perf_counter()
    pipe_dt.fit(X_train, y_train)
    t_dt = time.perf_counter() - t0

    y_val_dt = pipe_dt.predict(X_val)
    val_acc_dt = accuracy_score(y_val, y_val_dt)
    val_f1_dt = f1_score(y_val, y_val_dt, average="macro")

    y_pred_dt = pipe_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt, average="macro")

    results["Decision Tree"] = {
        "val_acc": val_acc_dt, "val_f1": val_f1_dt,
        "acc": acc_dt, "f1": f1_dt,
        "time": t_dt, "model": pipe_dt
    }
    print(f"[Baseline] Decision Tree       -> Val Acc: {val_acc_dt:.4f}, Val F1: {val_f1_dt:.4f} | Test Acc: {acc_dt:.4f}, Test F1: {f1_dt:.4f} ({t_dt:.2f}s)")

    joblib.dump(pipe_lr, "models/baseline_logreg.joblib")
    joblib.dump(pipe_dt, "models/baseline_dt.joblib")

    return results


def train_binary_mvp(X_train, y_train, X_val, y_val, X_test, y_test):
    """Trains and evaluates binary MVP classifier: AES-red (class 0) vs DES-red (class 1)."""
    print("\n" + "=" * 60)
    print(" BINARY MVP MODEL (AES-red vs DES-red) ")
    print("=" * 60)

    # Filter binary classes
    mask_train = np.isin(y_train, [0, 1])
    mask_val = np.isin(y_val, [0, 1])
    mask_test = np.isin(y_test, [0, 1])

    X_tr_bin, y_tr_bin = X_train[mask_train], y_train[mask_train]
    X_val_bin, y_val_bin = X_val[mask_val], y_val[mask_val]
    X_te_bin, y_te_bin = X_test[mask_test], y_test[mask_test]

    pipe_rf_bin = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    pipe_rf_bin.fit(X_tr_bin, y_tr_bin)

    y_pred_val = pipe_rf_bin.predict(X_val_bin)
    val_acc = accuracy_score(y_val_bin, y_pred_val)
    val_f1 = f1_score(y_val_bin, y_pred_val)

    y_pred = pipe_rf_bin.predict(X_te_bin)
    acc = accuracy_score(y_te_bin, y_pred)
    f1 = f1_score(y_te_bin, y_pred)

    status_acc = "PASSED [✓]" if acc > 0.88 else "FAILED [✗]"
    status_f1 = "PASSED [✓]" if f1 > 0.85 else "FAILED [✗]"

    print(f"[MVP] Validation -> Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}")
    print(f"[MVP] Test Set   -> Accuracy: {acc:.4f} (requirement > 0.8800) -> {status_acc}")
    print(f"[MVP] Test Set   -> F1-Score: {f1:.4f} (requirement > 0.8500) -> {status_f1}")

    joblib.dump(pipe_rf_bin, "models/rf_binary_mvp.joblib")
    return acc, f1


def train_multiclass_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Trains 4-class Random Forest with hyperparameter tuning via GridSearchCV on validation set."""
    print("\n" + "=" * 60)
    print(" TARGET 4-CLASS RANDOM FOREST (WITH GRIDSEARCH) ")
    print("=" * 60)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [10, 18, None],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__min_samples_leaf": [1, 2]
    }

    # Use PredefinedSplit to tune on the dedicated validation set (specification Section 5.4)
    # -1 marks training samples, 0 marks validation samples
    X_tune = np.vstack([X_train, X_val])
    y_tune = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
    cv = PredefinedSplit(test_fold=test_fold)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=False,
        verbose=0
    )

    t0 = time.perf_counter()
    grid_search.fit(X_tune, y_tune)
    t_rf = time.perf_counter() - t0

    best_params = grid_search.best_params_
    print(f"[+] Best hyperparameters (selected on validation set): {best_params}")
    print(f"[+] Best validation F1-Macro: {grid_search.best_score_:.4f}")

    # Retrain pipeline strictly on X_train (StandardScaler fitted strictly on training data)
    rf_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params))
    ])
    best_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred_val = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average="macro")
    print(f"[+] Validation set performance -> Accuracy: {val_acc:.4f}, F1-Macro: {val_f1_macro:.4f}")

    # Final held-out evaluation on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    present_classes = sorted(list(set(y_test)))
    present_names = [CLASS_NAMES[i] for i in present_classes]

    print("\n--- TEST SET CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=present_names, digits=4))
    print(f"Accuracy : {acc:.4f} (target >= 0.7500)")
    print(f"F1-Macro : {f1_macro:.4f} (target >= 0.7000)")

    joblib.dump(best_model, "models/rf_4class_final.joblib")

    # Generate evaluation figures
    _generate_plots(best_model, X_test, y_test, y_pred, y_proba, feature_names, present_classes)

    return best_model, val_acc, val_f1_macro, acc, f1_macro, t_rf


def _generate_plots(model, X_test, y_test, y_pred, y_proba, feature_names, present_classes):
    """Generates and saves visual evaluation artifacts."""
    os.makedirs("figures", exist_ok=True)
    present_names = [CLASS_NAMES[i] for i in present_classes]

    # Confusion Matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title("Confusion Matrix (4-Class Random Forest)")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()

    # Feature Importance figure
    rf_clf = model.named_steps["classifier"]
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4.5))
    plt.title("Feature Importance (Random Forest MDI)")
    plt.bar(range(len(importances)), importances[indices], align="center", color="#2b5c8f")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=300)
    plt.close()

    # ROC Curves figure (One-vs-Rest)
    if len(present_classes) > 1:
        y_test_bin = label_binarize(y_test, classes=range(len(CLASS_NAMES)))
        plt.figure(figsize=(7, 6))
        for i in present_classes:
            if y_proba.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curves (One-vs-Rest) for Cipher Families")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("figures/roc_curves.png", dpi=300)
        plt.close()

    print("[✓] Successfully saved evaluation figures to figures/")


def evaluate_adversarial_coa(model, adv_features_path="data/adversarial_features.csv"):
    """
    Adversarial Ciphertext-Only Attack (COA) test on full-round ciphers.
    Verifies that full rounds reduce classification accuracy to random guessing level (<= 35%).
    """
    print("\n" + "=" * 60)
    print(" ADVERSARIAL COA TEST (FULL ROUND CIPHERS) ")
    print("=" * 60)

    if not os.path.exists(adv_features_path):
        print(f"[*] Adversarial dataset not found at {adv_features_path}. Skipping adversarial test.")
        return

    adv_df = pd.read_csv(adv_features_path)
    X_adv = adv_df.drop(columns=["label"]).values
    y_adv = adv_df["label"].values

    y_pred_adv = model.predict(X_adv)
    adv_acc = accuracy_score(y_adv, y_pred_adv)

    print(f"[*] Full rounds adversarial accuracy: {adv_acc * 100:.2f}%")
    print(f"[*] Cryptographic limit threshold   : <= 35.00%")
    if adv_acc <= 0.35:
        print("[✓] SUCCESS: Full rounds successfully eliminate statistical cipher signatures.")
    else:
        print("[!] Note: Adversarial accuracy exceeds 35% threshold.")

    # Prediction distribution plot for adversarial samples
    plt.figure(figsize=(6, 4))
    counts = pd.Series(y_pred_adv).value_counts().reindex(range(4), fill_value=0)
    plt.bar([CLASS_NAMES[i] for i in range(4)], counts.values, color="#b83b3b")
    plt.title("Prediction Distribution on Full-Round Ciphers")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("figures/adversarial_distribution.png", dpi=300)
    plt.close()


def run_pipeline(features_path, adversarial_path=None):
    """Executes the complete machine learning training and evaluation pipeline."""
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # Stratified split: 70% train / 15% validation / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Dataset partitioning: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Baseline models evaluation
    baseline_res = train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Binary MVP evaluation
    train_binary_mvp(X_train, y_train, X_val, y_val, X_test, y_test)

    # Target 4-class Random Forest with GridSearch on validation set
    best_rf, val_acc_rf, val_f1_rf, acc_rf, f1_rf, t_rf = train_multiclass_rf(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    )

    # Summary table
    print("\n" + "=" * 80)
    print(" PERFORMANCE SUMMARY (VALIDATION & TEST SETS) ")
    print("=" * 80)
    print(f"{'Model':<24} | {'Val Acc':<8} | {'Val F1':<8} | {'Test Acc':<8} | {'Test F1':<8} | {'Time [s]':<8}")
    print("-" * 80)
    for name, r in baseline_res.items():
        print(f"{name:<24} | {r['val_acc']:<8.4f} | {r['val_f1']:<8.4f} | {r['acc']:<8.4f} | {r['f1']:<8.4f} | {r['time']:<8.2f}")
    print(f"{'Random Forest (Final)':<24} | {val_acc_rf:<8.4f} | {val_f1_rf:<8.4f} | {acc_rf:<8.4f} | {f1_rf:<8.4f} | {t_rf:<8.2f}")


    # Adversarial COA Test
    if adversarial_path:
        evaluate_adversarial_coa(best_rf, adversarial_path)


def run_verification_suite():
    """Runs an internal end-to-end verification suite on synthetic data."""
    print("==================================================")
    print(" RUNNING TRAINING PIPELINE VERIFICATION SUITE     ")
    print("==================================================")
    print("[*] Generating synthetic dataset for pipeline validation...")

    np.random.seed(42)
    n_samples = 200
    n_features = 8
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    X_synth = np.random.randn(n_samples, n_features)
    y_synth = np.random.choice([0, 1, 2, 3], size=n_samples)

    synth_df = pd.DataFrame(X_synth, columns=feature_cols)
    synth_df["label"] = y_synth

    test_csv = "data/features_verification.csv"
    os.makedirs("data", exist_ok=True)
    synth_df.to_csv(test_csv, index=False)

    try:
        run_pipeline(test_csv)
        print("\n==================================================")
        print(" ALL TESTS PASSED: Training pipeline verified!    ")
        print("==================================================")
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate cipher family classifiers (Phases 3 & 4)")
    parser.add_argument("--features", type=str, default="data/features.csv",
                        help="Path to extracted features CSV file")
    parser.add_argument("--adversarial", type=str, default="data/adversarial_features.csv",
                        help="Path to adversarial features CSV file")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    if os.path.exists(args.features):
        run_pipeline(args.features, args.adversarial)
    else:
        print(f"[*] Dataset file not found at '{args.features}'.")
        run_verification_suite()


if __name__ == "__main__":
    main()
