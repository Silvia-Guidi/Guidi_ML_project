import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.linear import LinearSVM, LogisticRegression
from models.kernel import KernelSVM, KernelLogisticRegression
from utils.evaluation import cross_val_score, cross_val_score_kernel, train_and_evaluate
from plots.plots import plot_metrics 

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
feature_columns = df.drop(columns=["quality"]).columns
for col in feature_columns:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)
skewed_features = ["residual_sugar", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides"]
for col in skewed_features:
    df[col] = np.log1p(df[col]) 
feature_names = df.drop(columns=["quality"]).columns
target = np.where(df["quality"] >= 6, 1, -1)
features = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state= 42, stratify= target
)
mean_train= features_train.mean(axis=0)
std_train = features_train.std(axis=0)
std_train[std_train == 0] = 1
features_train = (features_train - mean_train) / std_train
features_test = (features_test - mean_train) / std_train

num_points = 6
lambdas = np.logspace(-4, 0, num=num_points)
gammas  = np.logspace(-3, 1, num=num_points)

# Linear SVM 
best_lambda_svm, svm_results = cross_val_score(
    LinearSVM, features_train, target_train, lambdas, k=5,
    epochs=15, batch_size=64, shuffle=True, random_state=42
)

# Linear Logistic Regressions
best_lambda_lr, lr_results = cross_val_score(
    LogisticRegression, features_train, target_train, lambdas, k=5,
    epochs=50, batch_size=64, eta=0.1, shuffle=True, random_state=42
)
# final model train and evaluation
svm_model, svm_train_metrics, svm_test_metrics = train_and_evaluate(
    LinearSVM, features_train, target_train, features_test, target_test, best_lambda_svm
)
print(f"Linear SVM Test Accuracy: {svm_test_metrics['accuracy']:.4f}")
lr_model, lr_train_metrics, lr_test_metrics = train_and_evaluate(
    LogisticRegression, features_train, target_train, features_test, target_test, best_lambda_lr
)
print(f"Logistic Regression Test Accuracy: {lr_test_metrics['accuracy']:.4f}")

# Kernel SVM
(best_lambda_ksvm, best_degree_ksvm), kernel_svm_results = cross_val_score_kernel(
    KernelSVM, features_train, target_train, 
    lambdas, kernel_params=[2,3,4], k=5, kernel_param_name="degree",
    epochs=15, eta=0.1, coef0=1
)
# Kernel Logistic Regression
(best_lambda_klr, best_gamma_klr), kernel_lr_results = cross_val_score_kernel(
    KernelLogisticRegression, features_train, target_train, lambdas, gammas, k=5,
    epochs=50, eta=0.1
)

# final model train and evaluation
ksvm_model, ksvm_train_metrics, ksvm_test_metrics = train_and_evaluate(
    KernelSVM, features_train, target_train, features_test, target_test,
    best_lambda_ksvm, degree=best_degree_ksvm, epochs=15, eta=0.1, coef0=1
)
print(f"Kernel SVM Test Accuracy: {ksvm_test_metrics['accuracy']:.4f}")
klr_model, klr_train_metrics, klr_test_metrics = train_and_evaluate(
    KernelLogisticRegression, features_train, target_train, features_test, target_test,
    best_lambda_klr, gamma=best_gamma_klr, epochs=50, eta=0.1
)
print(f"Kernel Logistic Regression Test Accuracy: {klr_test_metrics['accuracy']:.4f}")

#plot
models_metrics = {
    "Linear SVM": (svm_train_metrics, svm_test_metrics),
    "Logistic Regression": (lr_train_metrics, lr_test_metrics),
    "Kernel SVM": (ksvm_train_metrics, ksvm_test_metrics),
    "Kernel Logistic Regression": (klr_train_metrics, klr_test_metrics)
}
plot_metrics(models_metrics)
