import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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

## SVM 
class LinearSVM:
    def __init__(self, lambda_reg=1e-3, epochs=15, batch_size=64, shuffle=True, random_state=42):
        self.lambda_reg = float(lambda_reg)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.w = None
        self.b = 0.0
        self.history_ = {"obj": []} 
    
    def _hingelosses(self, X, y):
        margins = 1 - y * (X @ self.w + self.b)
        hinge = np.maximum(0.0, margins).mean()
        reg = 0.5 * self.lambda_reg * (self.w @ self.w)
        return reg + hinge
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state) # random choice of the element
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)  #init the weight and bias vectors
        self.b = 0.0

        t = 0
        idx = np.arange(n)

        for epoch in range(self.epochs):  
            if self.shuffle:
                rng.shuffle(idx)

            for start in range(0, n, self.batch_size):  #update epoch and reg term
                t += 1
                eta = 1.0 / (self.lambda_reg * t)

                batch = idx[start:start + self.batch_size]  #get the example
                Xb = X[batch]
                yb = y[batch]

                # find violators: y*(w·x + b) < 1
                margins = yb * (Xb @ self.w + self.b)
                viol = margins < 1.0
                m = max(1, viol.sum())  # avoid divide-by-zero

                # shrink w for regularization
                self.w *= (1.0 - eta * self.lambda_reg)

                # subgradient step from violators
                if viol.any():
                    Xv = Xb[viol]
                    yv = yb[viol]
                    self.w += (eta / m) * (Xv.T @ yv)
                    # bias gets NO regularization
                    self.b += (eta / m) * yv.sum()
            
            # compute hinge loss on full dataset at end of epoch
            self.history_["obj"].append(self._hingelosses(X, y))

        return self
    
    def decision_function(self, X):
        return X @ self.w + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
    def score(self, X, y):
        return (self.predict(X) == y).mean()

# Logistic Regression
class LogisticRegression:
    def __init__(self, lambda_reg=1e-3, epochs=50, batch_size=64, eta=0.1,
                 shuffle=True, random_state=42):
        self.lambda_reg = float(lambda_reg)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.eta = float(eta)  # learning rate
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.w = None
        self.b = 0.0
        self.history_ = {"loss": []}  # track loss per epoch

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, X, y):
        z = X @ self.w + self.b
        # logistic loss
        log_loss = np.log(1 + np.exp(-y * z)).mean()
        # L2 regularization
        reg = 0.5 * self.lambda_reg * (self.w @ self.w)
        return log_loss + reg

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0

        idx = np.arange(n)

        for epoch in range(self.epochs):
            if self.shuffle:
                rng.shuffle(idx)

            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb = X[batch]
                yb = y[batch]

                z = Xb @ self.w + self.b
                p = self._sigmoid(-yb * z)  # σ(-y*(w·x+b))

                # gradient computation
                grad_w = -(Xb.T @ (yb * p)) / len(yb) + self.lambda_reg * self.w
                grad_b = -(yb * p).sum() / len(yb)

                # gradient descent update
                self.w -= self.eta * grad_w
                self.b -= self.eta * grad_b

            # track full loss per epoch
            self.history_["loss"].append(self._loss(X, y))

        return self

    def decision_function(self, X):
        return X @ self.w + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
    def score(self, X, y):
        return (self.predict(X) == y).mean()

def cross_val_score(model_class, X, y, lambdas, k=5, **model_params):
    results = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambdas:
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(lambda_reg=lam, **model_params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        results[lam] = np.mean(scores)
    best_lambda = max(results, key=results.get)
    return best_lambda, results

def train_and_evaluate(model_class, X_train, y_train, X_test, y_test, best_lambda, **model_params):
    model = model_class(lambda_reg=best_lambda, **model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    acc = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

    return model, metrics

# get the lambda
num_points = 6
lambdas = np.logspace(-4, 0, num=num_points)

# SVM 
best_lambda_svm, svm_results = cross_val_score(
    LinearSVM, features_train, target_train, lambdas, k=5,
    epochs=15, batch_size=64, shuffle=True, random_state=42
)

# Logistic Regression
best_lambda_lr, lr_results = cross_val_score(
    LogisticRegression, features_train, target_train, lambdas, k=5,
    epochs=50, batch_size=64, eta=0.1, shuffle=True, random_state=42
)

# final model train and evaluation
svm_model, svm_metrics = train_and_evaluate(
    LinearSVM, features_train, target_train, features_test, target_test,
    best_lambda_svm, epochs=15, batch_size=64, shuffle=True, random_state=42
)

lr_model, lr_metrics = train_and_evaluate(
    LogisticRegression, features_train, target_train, features_test, target_test,
    best_lambda_lr, epochs=50, batch_size=64, eta=0.1, shuffle=True, random_state=42
)

# plots
def plot_test_metrics(svm_metrics, lr_metrics):
    # Metrics to plot
    metric_names = ["accuracy", "precision", "recall", "f1"]
    
    svm_values = [svm_metrics[m] for m in metric_names]
    lr_values  = [lr_metrics[m] for m in metric_names]
    
    x = np.arange(len(metric_names))  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, svm_values, width, label="SVM")
    ax.bar(x + width/2, lr_values, width, label="Logistic Regression")

    ax.set_ylabel("Score")
    ax.set_title("Final Test Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.legend()
    ax.set_ylim(0, 1.05)  # scores between 0 and 1

    plt.tight_layout()
    plt.show()
plot_test_metrics(svm_metrics, lr_metrics)

def plot_confusion_matrices(svm_metrics, lr_metrics, class_labels=[1, -1]):
    cm_svm = svm_metrics["confusion_matrix"]
    cm_lr = lr_metrics["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cm, title in zip(axes, [cm_svm, cm_lr], ["SVM", "Logistic Regression"]):
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        # Show values inside cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")

    fig.colorbar(im[0], ax=axes, location="right", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(svm_metrics, lr_metrics)

def plot_training_curves(svm_model, lr_model):
    plt.figure(figsize=(10,4))

    # SVM
    plt.plot(svm_model.history_["obj"], label="SVM (hinge loss)")
    # LR
    plt.plot(lr_model.history_["loss"], label="LR (log loss)")
    
    plt.xlabel("Epoch")
    plt.ylabel("Objective / Loss")
    plt.title("Training Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_curves(svm_model, lr_model)

def plot_cv_results(svm_results, lr_results):
    plt.figure(figsize=(8,5))
    plt.semilogx(list(svm_results.keys()), list(svm_results.values()), 
                 marker="o", label="SVM")
    plt.semilogx(list(lr_results.keys()), list(lr_results.values()), 
                 marker="o", label="Logistic Regression")
    
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("CV Accuracy")
    plt.title("Cross-Validation Results")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.show()

plot_cv_results(svm_results, lr_results)

