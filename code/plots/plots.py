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
        bars_train = ax.bar(x - width/2, train_values[i], width, label="Train", alpha=0.7, color="#7ae582")
        bars_test = ax.bar(x + width/2, test_values[i], width, label="Test", alpha=0.7, color="#00a5cf")
        
        for bar in bars_train:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom')
        for bar in bars_test:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom')


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
        for j, cm in enumerate([train_metrics["confusion_matrix"], test_metrics["confusion_matrix"]]):
            # Convert counts to percentages per true class (row)
            cm_percent = cm.astype(np.float64)
            cm_percent = cm_percent / cm_percent.sum(axis=1, keepdims=True) * 100

            ax = axes[i, j]
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name} - {'Train' if j==0 else 'Test'} (%)")

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
    plt.title("Training Convergence linear models")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ktraining_curves(ksvm_model, klr_model):
    plt.figure(figsize=(10,4))
    # SVM
    plt.plot(ksvm_model.history_["hinge_loss"], label="SVM (hinge loss)")
    # LR
    plt.plot(klr_model.history_["log_loss"], label="LR (log loss)")
    
    plt.xlabel("Epoch")
    plt.ylabel("Objective / Loss")
    plt.title("Training Convergence kernel models")
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
        std_matrix  = np.zeros((len(lambdas), len(gammas)))
        for i, lam in enumerate(lambdas):
            for j, gam in enumerate(gammas):
                mean_matrix[i, j] = results[(lam, gam)][0]
                std_matrix[i, j]  = results[(lam, gam)][1]
        # Plot mean CV accuracy heatmap
        plt.figure(figsize=(8,6))
        im = plt.imshow(mean_matrix, origin="lower",
                        extent=(min(gammas), max(gammas), min(lambdas), max(lambdas)),
                        aspect="auto", cmap="viridis")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Gamma (log scale)")
        plt.ylabel("Lambda (log scale)")
        plt.title(f"{model_name} Kernelized Cross-Validation (Mean Accuracy)")
        plt.colorbar(im, label="Mean CV Accuracy")

        # Highlight best hyperparameters
        best_idx = np.unravel_index(np.argmax(mean_matrix), mean_matrix.shape)
        best_lam = lambdas[best_idx[0]]
        best_gam = gammas[best_idx[1]]
        plt.scatter(best_gam, best_lam, color="white", marker="*", s=200, edgecolors="k", label="Best")

        plt.legend()
        plt.tight_layout()
        plt.show()