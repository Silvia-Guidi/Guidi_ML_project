import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from models.linear import LinearSVM, LogisticRegression
from models.kernel import KernelSVM, KernelLogisticRegression
from utils.evaluation import cross_val_score, cross_val_score_kernel, train_and_evaluate
from plots.plots import plot_metrics, plot_confusion_matrices, plot_training_curves,plot_ktraining_curves, plot_cv_results, radar_misclass

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
# variable information
#print(wine_quality.variables)
#print(wine_quality.metadata)

#data exploration
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
print(df.isnull().sum())

# fig, axes = plt.subplots(3, 4, figsize=(16, 10))
# axes = axes.flatten()
# for i, col in enumerate(df.columns):
#     sns.histplot(df[col], kde=False, ax=axes[i], color="#2CEAA3")
#     axes[i].set_title(f"Distribution of {col}", fontsize=12)
# fig.suptitle("Distribution of the variables: histograms", fontsize=16)
# plt.tight_layout()
# plt.show()
# plt.close()

# fig, axes = plt.subplots(3, 4, figsize=(16,10))  # 3 rows, 4 columns
# axes = axes.flatten()
# for i, col in enumerate(df.columns):
#     axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor="#2CEAA3"))
#     axes[i].set_title(col)
# fig.suptitle("Distribition of the variables: boxplots", fontsize=16)
# plt.tight_layout()
# plt.show()
# plt.close()

# print(df.describe())
# print(df.corr())

##heatmap PLOT
# plt.figure(figsize=(8,15))
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# plt.title("Correlation Heatmap", fontsize=16, pad=20)
# plt.show()
# plt.close()

#data preprocessing

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

# n_features = features_train.shape[1]
# fig, axes = plt.subplots(1, n_features, figsize=(3*n_features, 5))
# for i in range(n_features):
#     axes[i].boxplot(features_train[:, i], patch_artist=True,
#                     boxprops=dict(facecolor="#2CEAA3"))
#     axes[i].set_title(feature_names[i], fontsize=7)
# plt.tight_layout()
# plt.show()
# plt.close()
print(target_train.min())
print(target_train.max())
print(np.isnan(features_train).any())
print(np.isnan(features_test).any())
print(features_train.mean(axis=0))
print(features_train.std(axis=0)) 

# parameters grids
lambdas = np.logspace(-4, 0, num=7)
gammas  = np.logspace(-3, 1, num=6)
degrees = [4, 5]

# Linear SVM 
best_lambda_svm, svm_results = cross_val_score(
    LinearSVM, features_train, target_train, lambdas, k=5,
    epochs=15, batch_size=64, shuffle=True, random_state=42
)
print(f"Best lambda Linear SVM: lambda={best_lambda_svm}")
# Linear Logistic Regression
best_lambda_lr, lr_results = cross_val_score(
    LogisticRegression, features_train, target_train, lambdas, k=5,
    epochs=50, batch_size=64, eta=0.1, shuffle=True, random_state=42
)
print(f"Best lambda Linear LR: lambda={best_lambda_lr}")
# final model train and evaluation
svm_model, svm_train_metrics, svm_test_metrics = train_and_evaluate(
    LinearSVM, features_train, target_train, features_test, target_test,
    lambda_reg=best_lambda_svm, epochs=50, batch_size=64
)
print(f"Linear SVM Test Accuracy: {svm_test_metrics['accuracy']:.4f}")
lr_model, lr_train_metrics, lr_test_metrics = train_and_evaluate(
    LogisticRegression, features_train, target_train, features_test, target_test, best_lambda_lr, 
    epochs=50, batch_size=64
)
print(f"Linear Logistic Regression Test Accuracy: {lr_test_metrics['accuracy']:.4f}")

# Gaussian Kernel SVM 
best_params_gksvm, gksvm_results = cross_val_score_kernel(
    KernelSVM, features_train, target_train,
    lambdas=lambdas, gammas=gammas,
    k=5, kind="gamma", epochs=100, batch_size=128
)
print(f"Best parameters Gaussian Kernel SVM: lambda={best_params_gksvm[0]}, gamma={best_params_gksvm[1]}")

gksvm_model, gksvm_train_metrics, gksvm_test_metrics = train_and_evaluate(
    KernelSVM, features_train, target_train, features_test, target_test, 
    lambda_reg=best_params_gksvm[0], gamma=best_params_gksvm[1],
    kind="gamma", epochs=100, batch_size=128
)
print(f"Gaussian Kernel SVM Test Accuracy: {gksvm_test_metrics['accuracy']:.4f}")

# Gaussian Kernel Logistic Regression
best_params_gklr, gklr_results = cross_val_score_kernel(
    KernelLogisticRegression, features_train, target_train, eta=0.1,
    lambdas=lambdas, gammas=gammas, k=5, kind="gamma", epochs=150, batch_size=128
)
print(f"Best parameters Gaussian Kernel LR: lambda={best_params_gklr[0]}, gamma={best_params_gklr[1]}")

gklr_model, gklr_train_metrics, gklr_test_metrics = train_and_evaluate(
    KernelLogisticRegression, features_train, target_train, features_test, target_test,
    lambda_reg=best_params_gklr[0], gamma=best_params_gklr[1],
    eta=0.1, kind="gamma", epochs=50, batch_size=128
)
print(f"Gaussian Kernel Logistic Regression Test Accuracy: {gklr_test_metrics['accuracy']:.4f}")

# Polynomial Kernel SVM
best_params_pksvm, pksvm_results = cross_val_score_kernel(
    KernelSVM, features_train, target_train,
    lambdas=lambdas, gammas=gammas, degrees=degrees,
    k=5, kind="poly", epochs=100, batch_size=128
)
print(f"Best parameters Polynomial Kernel SVM: lambda={best_params_pksvm[0]}, gamma={best_params_pksvm[1]}, degree={best_params_pksvm[2]}")

pksvm_model, pksvm_train_metrics, pksvm_test_metrics = train_and_evaluate(
    KernelSVM, features_train, target_train, features_test, target_test,
    lambda_reg=best_params_pksvm[0], gamma=best_params_pksvm[1], degree=best_params_pksvm[2],
    kind="poly", epochs=100, batch_size=128
)
print(f"Polynomial Kernel SVM Test Accuracy: {pksvm_test_metrics['accuracy']:.4f}")

# Polynomial Kernel Logistic Regression
best_params_pklr, pklr_results = cross_val_score_kernel(
    KernelLogisticRegression, features_train, target_train, eta=0.1,
    lambdas=lambdas, gammas=gammas, degrees=degrees,
    k=5, kind="poly", epochs=100, batch_size=128
)
print(f"Best parameters Polynomial Kernel LR: lambda={best_params_pklr[0]}, gamma={best_params_pklr[1]}, degree={best_params_pklr[2]}")

pklr_model, pklr_train_metrics, pklr_test_metrics = train_and_evaluate(
    KernelLogisticRegression, features_train, target_train, features_test, target_test,
    lambda_reg=best_params_pklr[0], gamma=best_params_pklr[1], degree=best_params_pklr[2],
    eta=0.1, kind="poly", epochs=50, batch_size=128
)
print(f"Polynomial Kernel Logistic Regression Test Accuracy: {pklr_test_metrics['accuracy']:.4f}")

# plots
models_metrics = {
    "Linear SVM": (svm_train_metrics, svm_test_metrics),
    "Logistic Regression": (lr_train_metrics, lr_test_metrics),
    "Gaussian Kernel SVM": (gksvm_train_metrics, gksvm_test_metrics),
    "Gaussian Kernel Logistic Regression": (gklr_train_metrics, gklr_test_metrics),
    "Polynomial Kernel SVM": (pksvm_train_metrics, pksvm_test_metrics),
    "Polynomial Kernel Logistic Regression": (pklr_train_metrics, pklr_test_metrics)
}
#plot_metrics(models_metrics)
# plot_confusion_matrices(models_metrics, class_labels=[1, -1])
# plot_training_curves(svm_model, lr_model)
# plot_ktraining_curves(gksvm_model, gklr_model)
# plot_ktraining_curves(pksvm_model, pklr_model)
# plot_cv_results(svm_results, model_name="Linear SVM", param_type="linear")
# plot_cv_results(lr_results, model_name="Linear Logistic Regression", param_type="linear")
plot_cv_results(gksvm_results, model_name="Gaussian Kernel SVM", param_type="kernel")
plot_cv_results(gklr_results, model_name="Gaussian Kernel Logistic Regression", param_type="kernel")
plot_cv_results(pksvm_results, model_name="Polynomial Kernel SVM", param_type="kernel")
plot_cv_results(pklr_results, model_name="Polynomial Kernel Logistic Regression", param_type="kernel")
radar_misclass(svm_model, features_train, target_train, features_test, target_test, feature_names,
               model_name="Linear SVM", train_color='#2af5ff', test_color='#f40000')
radar_misclass(lr_model,  features_train, target_train, features_test, target_test, feature_names,
               model_name="Linear Logistic Regression", train_color='#9ef01a', test_color='#f40000')
radar_misclass(gksvm_model,  features_train, target_train, features_test, target_test, feature_names,
               model_name="Gaussian Kernel SVM", train_color='#28c2ff', test_color='#f40000')
radar_misclass(gklr_model,  features_train, target_train, features_test, target_test, feature_names,
               model_name="Gaussian Kernel LR", train_color='#70e000', test_color='#f40000')
radar_misclass(pksvm_model,  features_train, target_train, features_test, target_test, feature_names,
               model_name="Polynomial Kernel SVM", train_color='#60afff', test_color='#f40000')
radar_misclass(pklr_model,  features_train, target_train, features_test, target_test, feature_names,
               model_name="Polynomial Kernel LR", train_color='#38b000', test_color='#f40000')