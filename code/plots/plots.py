import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Metrics 
def plot_test_metrics(svm_metrics, lr_metrics):
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

# confusion matrices
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
    #fig.colorbar(im[0], ax=axes, location="right", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# training curves
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

#cv results
def plot_cv_results(results, model_name="Model", param_type="linear"):
    if param_type == "linear":
        lambdas, means, stds = zip(*[(lam, res[0], res[1]) for lam, res in results.items()])
        plt.errorbar(lambdas, means, yerr=stds, fmt="-o", label=model_name)
        plt.xscale("log")
        plt.xlabel("Lambda (log scale)")
        plt.ylabel("CV Accuracy")
        plt.title(f"{model_name} Cross-Validation (mean ± std)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        plt.tight_layout()
        plt.show()

    elif param_type == "kernel":
        lambdas = sorted(set(lam for lam, _ in results.keys()))
        gammas  = sorted(set(gam for _, gam in results.keys()))
        mean_matrix = np.zeros((len(lambdas), len(gammas)))
        for i, lam in enumerate(lambdas):
            for j, gam in enumerate(gammas):
                mean_matrix[i, j] = results[(lam, gam)][0]
        plt.figure(figsize=(7, 5))
        im = plt.imshow(mean_matrix, origin="lower", 
                        extent=(min(gammas), max(gammas), min(lambdas), max(lambdas)),
                        aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Mean CV Accuracy")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Gamma (log scale)")
        plt.ylabel("Lambda (log scale)")
        plt.title(f"{model_name} Kernelized Cross-Validation")
        plt.tight_layout()
        plt.show()