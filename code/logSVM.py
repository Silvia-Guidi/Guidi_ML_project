import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.linear import LinearSVM
from models.kernel import KernelSVM
from utils.evaluation import cross_val_score, cross_val_score_kernel, train_and_evaluate

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
for col in ["residual_sugar", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides"]:
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
# final model train and evaluation
svm_model, svm_metrics = train_and_evaluate(
    LinearSVM, features_train, target_train, features_test, target_test,
    best_lambda_svm, epochs=15, batch_size=64, shuffle=True, random_state=42
)
print(f"Linear SVM Test Accuracy: {svm_metrics['accuracy']:.4f}")
(best_lambda_ksvm, best_degree_ksvm), kernel_svm_results = cross_val_score_kernel(
    KernelSVM, features_train, target_train, 
    lambdas, kernel_params=[2,3,4], k=5, kernel_param_name="degree",
    epochs=15, eta=0.1, coef0=1
)
ksvm_model, ksvm_metrics = train_and_evaluate(
    KernelSVM, features_train, target_train, features_test, target_test,
    best_lambda_ksvm, degree=best_degree_ksvm, epochs=15, eta=0.1, coef0=1
)
print(f"Kernel SVM Test Accuracy: {ksvm_metrics['accuracy']:.4f}")