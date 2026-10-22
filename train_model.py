# TODO: Implement model training, hyperparameter optimization, and evaluation (Phases 3 & 4):
# - Stratified split: 70% Train, 15% Validation, 15% Test (seed=42)
# - Baseline models:
#     * Multinomial Logistic Regression (with StandardScaler inside Pipeline)
#     * Single Decision Tree
# - MVP Binary Model:
#     * AES-red vs DES-red (RandomForestClassifier, target: Acc > 88%, F1 > 0.85)
# - Target 4-Class Model:
#     * StratifiedKFold (5 splits) + GridSearchCV / Optuna for RandomForestClassifier
#     * Export final model pipeline to models/rf_4class_final.joblib
# - Artifacts & Evaluation:
#     * Confusion Matrix, ROC curves (One-vs-Rest), Feature Importance charts to figures/
# - Adversarial COA Test:
#     * Evaluate on full-round ciphers (AES-10r, DES-16r)
#     * Confirm accuracy degradation to random guessing level (<= 35%)
