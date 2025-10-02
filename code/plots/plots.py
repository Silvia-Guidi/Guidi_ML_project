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
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(svm_model.history_["obj"], label="SVM train")
    if "val_obj" in svm_model.history_:
        plt.plot(svm_model.history_["val_obj"], label="SVM val", linestyle="--")
    plt.plot(lr_model.history_["loss"], label="LR train")
    if "val_loss" in lr_model.history_:
        plt.plot(lr_model.history_["val_loss"], label="LR val", linestyle="--")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Objective")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "train_acc" in svm_model.history_:
        plt.plot(svm_model.history_["train_acc"], label="SVM train acc")
    if "val_acc" in svm_model.history_:
        plt.plot(svm_model.history_["val_acc"], label="SVM val acc", linestyle="--")
    if "train_acc" in lr_model.history_:
        plt.plot(lr_model.history_["train_acc"], label="LR train acc")
    if "val_acc" in lr_model.history_:
        plt.plot(lr_model.history_["val_acc"], label="LR val acc", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ktraining_curves(ksvm_model, klr_model):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(ksvm_model.history_["hinge_loss"], label="SVM train")
    if "val_loss" in ksvm_model.history_:
        plt.plot(ksvm_model.history_["val_loss"], label="SVM val", linestyle="--")
    plt.plot(klr_model.history_["log_loss"], label="LR train")
    if "val_loss" in klr_model.history_:
        plt.plot(klr_model.history_["val_loss"], label="LR val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Objective")
    plt.title("Training & Validation Loss (Kernel Models)")
    plt.legend()
    plt.subplot(1, 2, 2)
    if "train_acc" in ksvm_model.history_:
        plt.plot(ksvm_model.history_["train_acc"], label="SVM train acc")
    if "val_acc" in ksvm_model.history_:
        plt.plot(ksvm_model.history_["val_acc"], label="SVM val acc", linestyle="--")
    if "train_acc" in klr_model.history_:
        plt.plot(klr_model.history_["train_acc"], label="LR train acc")
    if "val_acc" in klr_model.history_:
        plt.plot(klr_model.history_["val_acc"], label="LR val acc", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy (Kernel Models)")
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

    else:
        key_len = len(next(iter(results.keys())))

        if key_len == 3:  # (lambda, gamma, degree)
            lambdas = sorted(set(lam for lam, _, _ in results.keys()))
            gammas  = sorted(set(gam for _, gam, _ in results.keys()))
            degrees = sorted(set(deg for _, _, deg in results.keys()))

            n_deg = len(degrees)
            fig, axes = plt.subplots(1, n_deg, figsize=(6*n_deg,5), squeeze=False)

            for idx, deg in enumerate(degrees):
                mean_matrix = np.zeros((len(lambdas), len(gammas)))
                for i, lam in enumerate(lambdas):
                    for j, gam in enumerate(gammas):
                        mean_matrix[i, j] = results[(lam, gam, deg)][0]

                ax = axes[0, idx]
                c = ax.imshow(mean_matrix, origin="lower", aspect="auto", cmap="viridis")

                ax.set_xticks(range(len(gammas)))
                ax.set_xticklabels(gammas)
                ax.set_yticks(range(len(lambdas)))
                ax.set_yticklabels(lambdas)

                ax.set_xlabel("Gamma")
                ax.set_ylabel("Lambda")
                ax.set_title(f"Degree {deg}")

                # annotate
                for i in range(len(lambdas)):
                    for j in range(len(gammas)):
                        ax.text(j, i, f"{mean_matrix[i,j]:.3f}",
                                ha="center", va="center",
                                color="white" if mean_matrix[i,j]<0.5 else "black",
                                fontsize=8)

                # highlight best
                best_idx = np.unravel_index(np.argmax(mean_matrix), mean_matrix.shape)
                ax.scatter(best_idx[1], best_idx[0], color="red", marker="*", s=200, edgecolors="k")

            fig.colorbar(c, ax=axes.ravel().tolist(), label="Mean CV Accuracy")
            plt.tight_layout()
            plt.show()

def radar_misclass(model, X_train, y_train, X_test, y_test, feature_names, model_name,
                   train_color='green', test_color='red', alpha_fill=0.25, max_examples=None):

    def avg_mis(X, y):
        y_pred = model.predict(X)
        mis_idx = np.where(y_pred != y)[0]
        if len(mis_idx) == 0:
            return None
        if max_examples is not None:
            mis_idx = mis_idx[:max_examples]
        return np.mean(X[mis_idx], axis=0)

    # Compute averages
    train_avg = avg_mis(X_train, y_train)
    test_avg = avg_mis(X_test, y_test)

    if train_avg is None and test_avg is None:
        print(f"No misclassifications for {model_name} in either set.")
        return

    labels = np.array(feature_names)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    if train_avg is not None:
        train_stats = np.concatenate((train_avg, [train_avg[0]]))
        ax.plot(angles, train_stats, 'o-', linewidth=2, label=f"{model_name} Train", color=train_color)
        ax.fill(angles, train_stats, alpha=alpha_fill, color=train_color)

    if test_avg is not None:
        test_stats = np.concatenate((test_avg, [test_avg[0]]))
        ax.plot(angles, test_stats, 'o--', linewidth=2, label=f"{model_name} Test", color=test_color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Misclassified Feature Profile - {model_name}")
    ax.legend(loc="upper right")
    plt.show()