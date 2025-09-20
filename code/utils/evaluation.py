import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
        results[lam] = (np.mean(scores), np.std(scores))
    best_lambda = max(results, key=results.get)
    return best_lambda, results

def cross_val_score_kernel(model_class, X, y, lambdas, gammas, k=5, **model_params):
    results={}
    kf= KFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambdas:
        for gamma in gammas:
            folders_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = model_class(lambda_reg=lam, gamma=gamma, **model_params)
                model.fit(X_train, y_train)
                folders_scores.append(model.score(X_val, y_val))
            results[(lam, gamma)] = (np.mean(folders_scores), np.std(folders_scores))
    best_lambda, best_gamma = max(results, key=lambda x: results[x][0])
    return (best_lambda, best_gamma), results

def train_and_evaluate(model_class, X_train, y_train, X_test, y_test, best_lambda, **model_params):
    model = model_class(lambda_reg=best_lambda, **model_params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Metrics on test set
    test_metrics = {
        "accuracy": (y_pred_test == y_test).mean(),
        "precision": precision_score(y_test, y_pred_test, pos_label=1),
        "recall": recall_score(y_test, y_pred_test, pos_label=1),
        "f1": f1_score(y_test, y_pred_test, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test, labels=[1, -1])
    }

    # Metrics on train set
    train_metrics = {
        "accuracy": (y_pred_train == y_train).mean(),
        "precision": precision_score(y_train, y_pred_train, pos_label=1),
        "recall": recall_score(y_train, y_pred_train, pos_label=1),
        "f1": f1_score(y_train, y_pred_train, pos_label=1),
        "confusion_matrix": confusion_matrix(y_train, y_pred_train, labels=[1, -1])
    }

    return  model, train_metrics, test_metrics

def misclass(model, X_test, y_test, feature_names, max_examples=5):
    y_pred = model.predict(X_test)
    mis_idx = np.where(y_pred != y_test)[0]

    print(f"Total misclassified examples: {len(mis_idx)}")
    print(f"Showing up to {max_examples} examples:\n")

    for i in mis_idx[:max_examples]:
        print(f"Index: {i}, True: {y_test[i]}, Pred: {y_pred[i]}")
        features_str = ", ".join([f"{name}={X_test[i, j]:.2f}" for j, name in enumerate(feature_names)])
        print(f"  Features: {features_str}\n")
