import numpy as np
from utils.evaluation import val_metrics

def kernel(X, Y, kind="gamma", gamma=0.1, degree=3, coef0=1):
    X = np.atleast_2d(X).astype(np.float32)
    Y = np.atleast_2d(Y).astype(np.float32)
    if kind == "gamma":
        X_norm = np.sum(X**2, axis=1)[:, None]
        Y_norm = np.sum(Y**2, axis=1)[None, :]
        sq_dists = X_norm + Y_norm - 2 * (X @ Y.T)
        return np.exp(-gamma * sq_dists)
    elif kind == "poly":
        return (gamma * (X @ Y.T) + coef0) ** degree


class KernelLogisticRegression:
    def __init__(self, lambda_reg=0.1, gamma=0.1, epochs=50, eta=0.1,
                 batch_size=None, kind="gamma", degree=None, coef0=1):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.eta = float(eta)
        self.batch_size = batch_size
        self.kind = kind
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.history_ = {"log_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _stable_sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val=None, y_val=None, K_train=None):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X 
        self.alpha = np.zeros(n, dtype=np.float64)
        y01 = (y + 1) / 2  # Map {-1,1} -> {0,1}
        self.b = 0.0

        K_train = kernel(X, X, kind=self.kind, gamma=self.gamma,
                         degree=self.degree, coef0=self.coef0)
        K_train[np.diag_indices(n)] += 1e-12
        np.clip(K_train, 0, 1e10, out=K_train)  
        self.K_train = K_train

        for epoch in range(1, self.epochs + 1):
            f_no_bias = self.K_train @ self.alpha
            f = f_no_bias + self.b
            p = self._stable_sigmoid(f)

            grad_alpha = (self.K_train @ (p - y01)) / n + self.lambda_reg * f_no_bias
            grad_b = (p - y01).mean()
            # update
            self.alpha -= self.eta * grad_alpha
            np.clip(self.alpha, -1e6, 1e6, out=self.alpha)  
            self.b -= self.eta * grad_b

            # update history
            loss = -np.mean(y01 * np.log(p + 1e-12) + (1 - y01) * np.log(1 - p + 1e-12))
            reg_term = np.dot(self.alpha, self.K_train @ self.alpha) 
            reg_term = np.clip(reg_term, -1e12, 1e12)
            loss += 0.5 * self.lambda_reg * reg_term
            self.history_["log_loss"].append(float(loss))

            train_acc = (self.predict(X) == y).mean()
            self.history_["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = val_metrics(self, X_val, y_val, kind="kernel_logreg")
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

        return self

    def decision_function(self, X):
        X = np.atleast_2d(X)
        K_test = kernel(X, self.X_train, kind=self.kind, gamma=self.gamma,
                        degree=getattr(self, "degree", 3))
        return K_test @ self.alpha + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

class KernelSVM:
    def __init__(self, lambda_reg=0.01, gamma=0.1, epochs=100, batch_size=10,
                 random_state=42, kind="gamma", degree=None, coef0=1):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.kind = kind
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.history_ = {"hinge_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def fit(self, X, y, X_val=None, y_val=None, K_train=None):
        rng = np.random.default_rng(self.random_state)
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X  
        self.y_train = y
        self.alpha = np.zeros(n, dtype=np.float64)
        self.history_ = {"hinge_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        if K_train is None:
            K_train = kernel(X, X, kind=self.kind, gamma=self.gamma,
                             degree=self.degree, coef0=self.coef0)
            K_train[np.diag_indices(n)] += 1e-12
        self.K_train = K_train

        t = 1
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                yb = y[batch_idx]
                K_b_all = self.K_train[batch_idx, :]
                f_b = K_b_all @ (self.alpha * self.y_train)

                viol_mask = (yb * f_b) < 1.0
                eta_t = 1.0 / (self.lambda_reg * t)

                if np.any(viol_mask):
                    update = np.zeros_like(self.alpha)
                    update[batch_idx[viol_mask]] = eta_t / np.sum(viol_mask)
                    self.alpha += update
                    np.clip(self.alpha, 0, 1.0 / (self.lambda_reg * n), out=self.alpha)
                t += 1

            alpha_y = self.alpha * self.y_train
            f_all = self.K_train @ alpha_y

            # Update history
            hinge = np.maximum(0, 1 - self.y_train * f_all).mean()
            reg = 0.5 * self.lambda_reg * np.dot(alpha_y, self.K_train @ alpha_y)
            self.history_["hinge_loss"].append(float(hinge + reg))

            train_acc = (self.predict(X) == y).mean()
            self.history_["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, val_acc = val_metrics(self, X_val, y_val, kind="kernel_svm")
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

        return self

    def decision_function(self, X):
        X = np.atleast_2d(X)
        K_test = kernel(X, self.X_train, kind=self.kind, gamma=self.gamma,
                        degree=getattr(self, "degree", 3))
        return K_test @ (self.alpha * self.y_train)

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)