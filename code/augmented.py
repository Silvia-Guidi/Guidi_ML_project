import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.linear import LinearSVM, LogisticRegression
from models.kernel import KernelSVM, KernelLogisticRegression
from utils.evaluation import cross_val_score, cross_val_score_kernel, train_and_evaluate
from plots.plots import plot_metrics, plot_confusion_matrices, plot_training_curves,plot_ktraining_curves, plot_cv_results

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)

feature_names = df.drop(columns=["quality"]).columns
target = np.where(df["quality"] >= 6, 1, -1)
features = df.drop(columns=["quality"]).to_numpy(dtype=np.float32)

# Split train/test (raw, unstandardized)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# Data Augmentation (minority class only)
minority_idx = np.where(target_train == -1)[0]
features_minority = features_train[minority_idx]
target_minority = target_train[minority_idx]
# Number of synthetic samples (e.g., double the minority class)
n_synth = len(features_minority)
# Standard deviation of each feature in training set (raw)
feature_std = features_train.std(axis=0)
# Set random seed for reproducibility
np.random.seed(42)
# Generate synthetic samples by adding small Gaussian noise
synthetic_features = features_minority + np.random.normal(
    loc=0,
    scale=0.01 * feature_std,
    size=(n_synth, features_train.shape[1])
)
synthetic_target = np.full(n_synth, -1)

# Combine synthetic samples with original training set
features_train_aug = np.vstack([features_train, synthetic_features])
target_train_aug = np.hstack([target_train, synthetic_target])

print(f"Original training set size: {features_train.shape[0]}")
print(f"Augmented training set size: {features_train_aug.shape[0]}")

# Check class distribution in augmented training set
classes, counts = np.unique(target_train_aug, return_counts=True)
proportions = counts / counts.sum() * 100
for cls, count, prop in zip(classes, counts, proportions):
    label = "Good (1)" if cls == 1 else "Bad (-1)"
    print(f"{label}: {count} samples, {prop:.1f}%")

# Standardization after augmentation
mean_train = features_train_aug.mean(axis=0)
std_train = features_train_aug.std(axis=0)
std_train[std_train == 0] = 1
features_train_aug = (features_train_aug - mean_train) / std_train
features_test = (features_test - mean_train) / std_train  # use train mean/std

num_points = 6
lambdas = np.logspace(-4, 0, num=num_points)
gammas  = np.logspace(-3, 1, num=num_points)

# Linear SVM 
best_lambda_svm, svm_results = cross_val_score(
    LinearSVM, features_train, target_train, lambdas, k=5,
    epochs=15, shuffle=True, random_state=42
)
print(f"Best lambda Linear augmented SVM: lambda={best_lambda_svm}")
# Linear Logistic Regressions
best_lambda_lr, lr_results = cross_val_score(
    LogisticRegression, features_train, target_train, lambdas, k=5,
    epochs=50, eta=0.1, shuffle=True, random_state=42
)
print(f"Best lambda Linear augmented LR: lambda={best_lambda_lr}")
# final model train and evaluation
svm_model, svm_train_metrics, svm_test_metrics = train_and_evaluate(
    LinearSVM, features_train_aug, target_train_aug, features_test, target_test, best_lambda_svm
)
print(f"Linear SVM Test Accuracy after augmentation: {svm_test_metrics['accuracy']:.4f}")
lr_model, lr_train_metrics, lr_test_metrics = train_and_evaluate(
    LogisticRegression, features_train_aug, target_train_aug, features_test, target_test, best_lambda_lr
)
print(f"Linear Logistic Regression Test Accuracy after augmentation: {lr_test_metrics['accuracy']:.4f}")


# Kernel SVM
(best_lambda_ksvm_aug, best_gamma_ksvm_aug), kernel_svm_results_aug = cross_val_score_kernel(
    KernelSVM, features_train_aug, target_train_aug,
    etas=0.1, lambdas=lambdas, gammas=gammas, k=5,
    epochs=15
)
print(f"Best parameters Kernel SVM augmented: lambda={best_lambda_ksvm_aug}, gamma={best_gamma_ksvm_aug}")
# Kernel Logistic Regression
(best_lambda_klr_aug, best_gamma_klr_aug, best_eta_klr_aug), kernel_lr_results_aug = cross_val_score_kernel(
    KernelLogisticRegression, features_train_aug, target_train_aug,
    etas=0.1, lambdas=lambdas, gammas=gammas, k=5,
    epochs=15
)
print(f"Best parameters Kernel LR augmented: lambda={best_lambda_klr_aug}, gamma={best_gamma_klr_aug}")
# final model train and evaluation
ksvm_model_aug, ksvm_train_metrics_aug, ksvm_test_metrics_aug = train_and_evaluate(
    KernelSVM, features_train_aug, target_train_aug, features_test, target_test,
    best_lambda_ksvm_aug, gamma=best_gamma_ksvm_aug, epochs=15, eta=0.1
)
print(f"Kernel SVM Test Accuracy after augmentation: {ksvm_test_metrics_aug['accuracy']:.4f}")

klr_model_aug, klr_train_metrics_aug, klr_test_metrics_aug = train_and_evaluate(
    KernelLogisticRegression, features_train_aug, target_train_aug, features_test, target_test,
    best_lambda_klr_aug, gamma=best_gamma_klr_aug, epochs=50, eta=0.1
)
print(f"Kernel Logistic Regression Test Accuracy after augmentation: {klr_test_metrics_aug['accuracy']:.4f}")
#plot
models_metrics = {
    "Linear SVM": (svm_train_metrics, svm_test_metrics),
    "Logistic Regression": (lr_train_metrics, lr_test_metrics),
    "Kernel SVM": (ksvm_train_metrics_aug, ksvm_test_metrics_aug),
    "Kernel Logistic Regression": (klr_train_metrics_aug, klr_test_metrics_aug)
}
plot_metrics(models_metrics)
plot_confusion_matrices(models_metrics, class_labels=[1, -1])
plot_training_curves(svm_model, lr_model)
plot_ktraining_curves(ksvm_model_aug, klr_model_aug)
plot_cv_results(svm_results, model_name="Linear SVM", param_type="linear")
plot_cv_results(lr_results, model_name="Linear Logistic Regression", param_type="linear")
plot_cv_results(kernel_svm_results_aug, model_name="Kernel SVM", param_type="kernel")
plot_cv_results(kernel_lr_results_aug, model_name="Kernel Logistic Regression", param_type="kernel")