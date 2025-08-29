import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Metrics 
def plot_metrics(models_metrics):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    model_names = list(models_metrics.keys())

    # Collect metrics into arrays
    train_values = np.array([[models_metrics[m][0][metric] for metric in metric_names] for m in model_names])
    test_values = np.array([[models_metrics[m][1][metric] for metric in metric_names] for m in model_names])

    x = np.arange(len(metric_names))  
    width = 0.35  
    fig, axes = plt.subplots(len(model_names), 1, figsize=(10, 5 * len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    for i, model in enumerate(model_names):
        ax = axes[i]
        ax.bar(x - width/2, train_values[i], width, label="Train", alpha=0.7)
        ax.bar(x + width/2, test_values[i], width, label="Test", alpha=0.7)

        ax.set_ylabel("Score")
        ax.set_title(f"{model} - Train vs Test Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metric_names])
        ax.legend()
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

# confusion matrices
def plot_confusion_matrices(models_metrics, class_labels=[1, -1]):
    num_models = len(models_metrics)
    fig, axes = plt.subplots(num_models, 2, figsize=(10, 4 * num_models))

    if num_models == 1:
        axes = np.array([axes])  # ensure consistency for single model

    for i, (model_name, (train_metrics, test_metrics)) in enumerate(models_metrics.items()):
        # Training confusion matrix
        sns.heatmap(train_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[i, 0])
        axes[i, 0].set_title(f"{model_name} - Train")
        axes[i, 0].set_xlabel("Predicted")
        axes[i, 0].set_ylabel("Actual")

        # Test confusion matrix
        sns.heatmap(test_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[i, 1])
        axes[i, 1].set_title(f"{model_name} - Test")
        axes[i, 1].set_xlabel("Predicted")
        axes[i, 1].set_ylabel("Actual")

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