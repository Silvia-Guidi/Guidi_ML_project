import numpy as np

# Kernel
def kernel(X, Y, gamma=0.1):
     X = X.astype(np.float64)
     Y = Y.astype(np.float64)
     X_norm = np.sum(X**2, axis=1)[:, None]
     Y_norm = np.sum(Y**2, axis=1)[None, :]
     sq_dists = X_norm + Y_norm - 2 * X @ Y.T
     return np.exp(-gamma * sq_dists)

class KernelLogisticRegression:
    def __init__(self, lambda_reg=0.1, gamma=0.1, epochs=100, eta=0.01, batch_size=None):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.eta = float(eta)
        self.batch_size = batch_size
        self.alpha = None
        self.b = 0.0
        self.X_train = None

    def _stable_sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X.copy()
        y01 = (y + 1) / 2  # Map {-1,1} -> {0,1}
        # Kernel with tiny jitter for stability
        K_train = kernel(X, X, self.gamma) + 1e-12 * np.eye(n, dtype=np.float64)
        self.alpha = np.zeros(n, dtype=np.float64)
        self.b = 0.0
        batch_size = self.batch_size or n  # full-batch if None

        for epoch in range(self.epochs):
            f = K_train @ self.alpha + self.b
            p = self._stable_sigmoid(f)
            grad_alpha = (K_train @ (p - y01)) / n
            grad_alpha += self.lambda_reg * (K_train @ self.alpha)
            grad_b = (p - y01).mean()
            self.alpha -= self.eta * grad_alpha
            self.b -= self.eta * grad_b
        return self

    def decision_function(self, X):
        X = X.astype(np.float64)
        K_test = kernel(X, self.X_train, self.gamma)
        f = K_test @ self.alpha + self.b
        return np.clip(f, -1e6, 1e6)  # numerical safety

    def predict(self, X):
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1)

    def score(self, X, y):
        y = y.astype(np.float64)
        return np.mean(self.predict(X) == y)


class KernelSVM:
    def __init__(self, lambda_reg=0.01, gamma=0.1, epochs=50, eta=0.1, **kwargs):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.eta = float(eta)
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.history_ = {"hinge_loss": []}

    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = kernel(X, X, self.gamma) + 1e-12 * np.eye(n)
        self.alpha = np.zeros(n, dtype=np.float64)
        self.b = 0.0
        for epoch in range(self.epochs):
            f = K @ (self.alpha * y) + self.b
            viol = (y * f) < 1
            # full-batch update for α
            grad_alpha = -K[:, viol] @ y[viol] / n + self.lambda_reg * (K @ self.alpha)
            self.alpha -= self.eta * grad_alpha
            # bias update: mean over violators
            if viol.any():
                self.b += self.eta * y[viol].mean()
            # track hinge loss
            f_all = K @ (self.alpha * y) + self.b
            loss = np.maximum(0, 1 - y * f_all).mean() + 0.5 * self.lambda_reg * (self.alpha @ K @ self.alpha)
            self.history_["hinge_loss"].append(float(loss))
        return self

    def decision_function(self, X):
        K_test = kernel(X, self.X_train, self.gamma)
        return K_test @ (self.alpha * self.y_train) + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)