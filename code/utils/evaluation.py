import numpy as np
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

def cross_val_score_kernel(model_class, X, y, lambdas, kernel_params, k=5, kernel_param_name="gamma", **model_params):
    results = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambdas:
        for kp in kernel_params:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = model_class(lambda_reg=lam, **{kernel_param_name: kp}, **model_params)
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))

            results[(lam, kp)] = (np.mean(scores), np.std(scores))

    best_lambda, best_kp = max(results, key=results.get)
    return (best_lambda, best_kp), results

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
