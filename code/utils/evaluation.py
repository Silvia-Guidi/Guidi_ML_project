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

def cross_val_score_kernel(model_class, X, y, etas, lambdas, gammas, 
                                  k=5, n_samples=30, refine_factor=2, **model_params):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    def evaluate(lam, gam, eta):
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = model_class(lambda_reg=lam, gamma=gam, eta=eta, **model_params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        return np.mean(scores), np.std(scores)
    
    param_grid = [(lam, gam, eta) for lam in lambdas for gam in gammas for eta in etas]
    sampled = random.sample(param_grid, min(n_samples, len(param_grid)))
    
    results = {}
    for lam, gam, eta in sampled:
        mean_score, std_score = evaluate(lam, gam, eta)
        results[(lam, gam, eta)] = (mean_score, std_score)

    best_lambda, best_gamma, best_eta = max(results, key=lambda x: results[x][0])
    
    def refine(values, best, factor):
        idx = values.index(best)
        lo = max(0, idx - factor)
        hi = min(len(values), idx + factor + 1)
        return values[lo:hi]
    
    refined_lambdas = refine(list(lambdas), best_lambda, refine_factor)
    refined_gammas = refine(list(gammas), best_gamma, refine_factor)
    refined_etas = refine(list(etas), best_eta, refine_factor)
    for lam in refined_lambdas:
        for gam in refined_gammas:
            for eta in refined_etas:
                if (lam, gam, eta) not in results:
                    mean_score, std_score = evaluate(lam, gam, eta)
                    results[(lam, gam, eta)] = (mean_score, std_score)
    best_lambda, best_gamma, best_eta = max(results, key=lambda x: results[x][0])
    return (best_lambda, best_gamma, best_eta), results

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
