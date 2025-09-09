import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from models.linear import LinearSVM, LogisticRegression
from models.kernel import KernelSVM, KernelLogisticRegression
from utils.evaluation import cross_val_score, cross_val_score_kernel, train_and_evaluate
from plots.plots import plot_metrics, plot_confusion_matrices, plot_training_curves,plot_ktraining_curves, plot_cv_results

from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
# variable information
#print(wine_quality.variables)
#print(wine_quality.metadata)

#data exploration
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
#print(df.isnull().sum())

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=False, ax=axes[i], color="#2CEAA3")
    axes[i].set_title(f"Distribution of {col}", fontsize=12)
fig.suptitle("Distribution of the variables: histograms", fontsize=16)
plt.tight_layout()
#plt.show()

fig, axes = plt.subplots(3, 4, figsize=(16,10))  # 3 rows, 4 columns
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor="#2CEAA3"))
    axes[i].set_title(col)
fig.suptitle("Distribition of the variables: boxplots", fontsize=16)
plt.tight_layout()
#plt.show()

#print(df.describe())
#print(df.corr())

##heatmap PLOT
plt.figure(figsize=(8,15))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap", fontsize=16, pad=20)
#plt.show()

#data preprocessing

feature_names = df.drop(columns=["quality"]).columns
target = np.where(df["quality"] >= 6, 1, -1)

classes, counts = np.unique(target, return_counts=True)
labels = ['Bad', ' Good']
colors = ['#FF9999', '#2CEAA3']
plt.figure(figsize=(5,5))
plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor':'k'})
plt.title("Proportion of Quality classes")
#plt.show()
features = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state= 42, stratify= target
)
mean_train= features_train.mean(axis=0)
std_train = features_train.std(axis=0)
std_train[std_train == 0] = 1
features_train = (features_train - mean_train) / std_train
features_test = (features_test - mean_train) / std_train

n_features = features_train.shape[1]
fig, axes = plt.subplots(1, n_features, figsize=(3*n_features, 5))
for i in range(n_features):
    axes[i].boxplot(features_train[:, i], patch_artist=True,
                    boxprops=dict(facecolor="#2CEAA3"))
    axes[i].set_title(feature_names[i], fontsize=7)
plt.tight_layout()
#plt.show()
print(np.isnan(features_train).any())
print(np.isnan(features_test).any())
print(features_train.mean(axis=0))
print(features_train.std(axis=0)) 

# get the lambda
num_points = 6
lambdas = np.logspace(-4, 0, num=num_points)
gammas  = np.logspace(-3, 1, num=num_points)

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
    LinearSVM, features_train, target_train, features_test, target_test, best_lambda_svm
)
print(f"Linear SVM Test Accuracy: {svm_test_metrics['accuracy']:.4f}")
lr_model, lr_train_metrics, lr_test_metrics = train_and_evaluate(
    LogisticRegression, features_train, target_train, features_test, target_test, best_lambda_lr
)
print(f"Linear Logistic Regression Test Accuracy: {lr_test_metrics['accuracy']:.4f}")

# Kernel SVM
(best_lambda_ksvm, best_gamma_ksvm), kernel_svm_results = cross_val_score_kernel(
    KernelSVM, features_train, target_train, etas=0.1,
    lambdas=lambdas, gammas=gammas, k=5,
    epochs=15
)
print(f"Best parameters Kernel SVM: lambda={best_lambda_ksvm}, gamma={best_gamma_ksvm}")
# Kernel Logistic Regression
(best_lambda_klr, best_gamma_klr), kernel_lr_results = cross_val_score_kernel(
    KernelLogisticRegression, features_train, target_train, eta=0.1,
    lambdas=lambdas, gammas=gammas, k=5,
    epochs=50
)
print(f"Best parameters Kernel LR: lambda={best_lambda_ksvm}, gamma={best_gamma_ksvm}")

# final model train and evaluation
ksvm_model, ksvm_train_metrics, ksvm_test_metrics = train_and_evaluate(
    KernelSVM, features_train, target_train, features_test, target_test,
    best_lambda_ksvm, gamma=best_gamma_ksvm, eta=0.1, epochs=15
)
print(f"Kernel SVM Test Accuracy: {ksvm_test_metrics['accuracy']:.4f}")

klr_model, klr_train_metrics, klr_test_metrics = train_and_evaluate(
    KernelLogisticRegression, features_train, target_train, features_test, target_test,
    best_lambda_klr, gamma=best_gamma_klr, eta=0.1, epochs=50
)
print(f"Kernel Logistic Regression Test Accuracy: {klr_test_metrics['accuracy']:.4f}")

# misclassification analysis
def misclass(model, X_test, y_test, feature_names, max_examples=5):
    y_pred = model.predict(X_test)
    mis_idx = np.where(y_pred != y_test)[0]

    print(f"Total misclassified examples: {len(mis_idx)}")
    print(f"Showing up to {max_examples} examples:\n")

    for i in mis_idx[:max_examples]:
        print(f"Index: {i}, True: {y_test[i]}, Pred: {y_pred[i]}")
        features_str = ", ".join([f"{name}={X_test[i, j]:.2f}" for j, name in enumerate(feature_names)])
        print(f"  Features: {features_str}\n")
#misclass(svm_model, features_test, target_test, feature_names)
#misclass(lr_model, features_test, target_test, feature_names)
#misclass(ksvm_model, features_test, target_test, feature_names)
#misclass(klr_model, features_test, target_test, feature_names)

# plots
models_metrics = {
    "Linear SVM": (svm_train_metrics, svm_test_metrics),
    "Logistic Regression": (lr_train_metrics, lr_test_metrics),
    "Kernel SVM": (ksvm_train_metrics, ksvm_test_metrics),
    "Kernel Logistic Regression": (klr_train_metrics, klr_test_metrics)
}
plot_metrics(models_metrics)
plot_confusion_matrices(models_metrics, class_labels=[1, -1])
plot_training_curves(svm_model, lr_model)
plot_ktraining_curves(ksvm_model, klr_model)
plot_cv_results(svm_results, model_name="Linear SVM", param_type="linear")
plot_cv_results(lr_results, model_name="Linear Logistic Regression", param_type="linear")
plot_cv_results(kernel_svm_results, model_name="Kernel SVM", param_type="kernel")
plot_cv_results(kernel_lr_results, model_name="Kernel Logistic Regression", param_type="kernel")