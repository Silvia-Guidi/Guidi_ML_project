import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def cross_val_score(model_class, X, y, lambdas, k=5, **model_params):
    results = {}
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for lam in lambdas:
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(lambda_reg=lam, **model_params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        results[lam] = (np.mean(scores), np.std(scores))
    best_lambda = max(results, key=results.get)
    return best_lambda, results

def cross_val_score_kernel(model_class, X, y, lambdas, gammas, degrees=None, k=5, kind="gamma", 
                           patience=5, max_epochs=None, **model_params):
    from models.kernel import kernel
    results = {}
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Degree iteration if polynomial kernel
    if kind == "poly":
        if degrees is None:
            raise ValueError("Degrees must be provided for polynomial kernel")
        degree_iter = degrees
    else:
        degree_iter = [None]

    # Precompute kernels for each (gamma, degree)
    precomputed_kernels = {}
    for gamma in gammas:
        for degree in degree_iter:
            K_full = kernel(X, X, kind=kind, gamma=gamma, degree=degree, coef0=model_params.get("coef0", 1))
            np.fill_diagonal(K_full, K_full.diagonal() + 1e-12)
            precomputed_kernels[(gamma, degree)] = K_full

    # Grid search over parameters
    for gamma in gammas:
        for degree in degree_iter:
            K_full = precomputed_kernels[(gamma, degree)]
            for lam in lambdas:

                def run_fold(train_idx, val_idx):
                    K_train = K_full[np.ix_(train_idx, train_idx)]
                    K_val   = K_full[np.ix_(val_idx, train_idx)]
                    model_kwargs = {
                        "lambda_reg": lam,
                        "gamma": gamma,
                        "kind": kind,
                        "degree": degree if kind == "poly" else None,
                        "patience": patience,          # pass early stopping
                        **model_params
                    }
                    if max_epochs is not None:
                        model_kwargs["epochs"] = max_epochs  # override epochs if needed

                    model = model_class(**model_kwargs)
                    model.fit(X[train_idx], y[train_idx],
                              X_val=X[val_idx], y_val=y[val_idx],
                              K_train=K_train)
                    return model.score(X[val_idx], y[val_idx])

                # run folds in parallel
                fold_scores = Parallel(n_jobs=-1)(
                    delayed(run_fold)(train_idx, val_idx)
                    for train_idx, val_idx in kf.split(X, y)
                )

                scores = np.array(fold_scores, dtype=np.float64)
                results[(lam, gamma, degree)] = (scores.mean(), scores.std())

    best_params = max(results, key=lambda x: results[x][0])
    return best_params, results
    
def train_and_evaluate(model_class, X_train, y_train, X_test, y_test, 
                       lambda_reg, gamma=None, degree=None, eta=None, 
                       epochs=50, batch_size=None, kind=None, coef0=None):
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model_kwargs = {"lambda_reg": lambda_reg, "epochs": epochs, "batch_size": batch_size}
    # Add optional kwargs only if they are not None
    if gamma is not None:
        model_kwargs["gamma"] = gamma
    if degree is not None:
        model_kwargs["degree"] = degree
    if eta is not None:
        model_kwargs["eta"] = eta
    if kind is not None:
        model_kwargs["kind"] = kind
    if coef0 is not None:
        model_kwargs["coef0"] = coef0
    model = model_class(**{k: v for k, v in model_kwargs.items() if v is not None})
    
    # Pass validation set for curve tracking
    model.fit(X_train_final, y_train_final, X_val=X_val, y_val=y_val)
    
    # Train metrics
    y_train_pred = model.predict(X_train_final)
    train_metrics = {
        "accuracy": (y_train_pred == y_train_final).mean(),
        "precision": precision_score(y_train_final, y_train_pred, pos_label=1),
        "recall": recall_score(y_train_final, y_train_pred, pos_label=1),
        "f1": f1_score(y_train_final, y_train_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_train_final, y_train_pred, labels=[1, -1])
    }
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_metrics = {
        "accuracy": (y_test_pred == y_test).mean(),
        "precision": precision_score(y_test, y_test_pred, pos_label=1),
        "recall": recall_score(y_test, y_test_pred, pos_label=1),
        "f1": f1_score(y_test, y_test_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=[1, -1])
    }
    return model, train_metrics, test_metrics

def val_metrics(model, X_val, y_val, kind="linear"):
    val_acc = np.mean(model.predict(X_val) == y_val)  # validation accuracy

    if kind == "linear":
        # Linear Logistic or Linear SVM
        if hasattr(model, "_loss"):  # logistic regression
            val_loss = model._loss(X_val, y_val)
        else:  # linear SVM
            val_loss = model._hingelosses(X_val, y_val)

    elif kind == "kernel_logreg":
        # Map labels {-1,1} -> {0,1}
        y_val01 = (y_val + 1) / 2
        f_val = model.decision_function(X_val)
        # stable sigmoid
        p_val = 1 / (1 + np.exp(-np.clip(f_val, -50, 50)))
        # regularized logistic loss
        log_loss = -np.mean(y_val01 * np.log(p_val + 1e-12) + (1 - y_val01) * np.log(1 - p_val + 1e-12))
        reg_term = 0.5 * model.lambda_reg * np.dot(model.alpha, model.K_train @ model.alpha)
        val_loss = float(log_loss + reg_term)

    elif kind == "kernel_svm":
        f_val = model.decision_function(X_val)
        # hinge loss
        hinge_loss = np.maximum(0, 1 - y_val * f_val).mean()
        reg_term = 0.5 * model.lambda_reg * np.dot(model.alpha * model.y_train, model.K_train @ (model.alpha * model.y_train))
        val_loss = float(hinge_loss + reg_term)

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return val_loss, val_acc