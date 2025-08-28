import matplotlib.pyplot as plt
import numpy as np

# Metrics 
def plot_test_metrics(svm_metrics, lr_metrics, ksvm_metrics=None, klr_metrics=None):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    
    # Build a dictionary of metrics
    metrics_dict = {
        "Linear SVM": svm_metrics,
        "Logistic Regression": lr_metrics
    }
    if ksvm_metrics is not None:
        metrics_dict["Kernel SVM"] = ksvm_metrics
    if klr_metrics is not None:
        metrics_dict["Kernel Logistic Regression"] = klr_metrics

    model_names = list(metrics_dict.keys())
    
    # Extract metric values for each model
    values = np.array([[metrics_dict[model][m] for m in metric_names] for model in model_names])
    x = np.arange(len(metric_names))  
    width = 0.15  

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot one bar group per model
    for i, model in enumerate(model_names):
        ax.bar(x + i*width - (len(model_names)-1)*width/2, values[i], width, label=model)

    ax.set_ylabel("Score")
    ax.set_title("Final Test Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

# confusion matrices
def plot_confusion_matrices(svm_metrics, lr_metrics, class_labels=[1, -1], cmap="coolwarm"):
    cm_svm = svm_metrics["confusion_matrix"]
    cm_lr = lr_metrics["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cm, title in zip(axes, [cm_svm, cm_lr], ["SVM", "Logistic Regression"]):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)  # use the chosen colormap
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
                text_color = "white" if cm[i, j] > cm.max() / 2. else "black"
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=text_color)

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