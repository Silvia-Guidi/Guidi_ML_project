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
    def __init__(self, lambda_reg=0.1, gamma=0.1, epochs=50, eta=0.1, batch_size=None):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.eta = float(eta)
        self.batch_size = batch_size
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.history_ = {"log_loss": []}

    def _stable_sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y,):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X.copy()
        y01 = (y + 1) / 2  # Map {-1,1} -> {0,1}
        K_train = kernel(X, X, self.gamma) + 1e-12 * np.eye(n)
        self.alpha = np.zeros(n, dtype=np.float64)
        self.b = 0.0
        self.history_ = {"log_loss": []}

        for epoch in range(1, self.epochs + 1):
            f = K_train @ self.alpha + self.b
            p = self._stable_sigmoid(f)
            # gradient
            grad_alpha = (K_train @ (p - y01)) / n + self.lambda_reg * (K_train @ self.alpha)
            grad_b = (p - y01).mean()
            # update
            self.alpha -= self.eta * grad_alpha
            self.b -= self.eta * grad_b
            # track log loss
            loss = -np.mean(y01 * np.log(p + 1e-12) + (1 - y01) * np.log(1 - p + 1e-12))
            loss += 0.5 * self.lambda_reg * (self.alpha @ K_train @ self.alpha)
            self.history_["log_loss"].append(float(loss))

        return self

    def decision_function(self, X):
        K_test = kernel(X, self.X_train, self.gamma)
        return K_test @ self.alpha + self.b

    def predict(self, X):
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class KernelSVM:
    def __init__(self, lambda_reg=0.01, gamma=0.1, epochs=100, batch_size=10, random_state=42):
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.batch_size=int(batch_size)
        self.random_state = int(random_state)
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.history_ = {"hinge_loss": []}

    def fit(self, X, y, K_train=None):
        rng = np.random.default_rng(self.random_state)
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.alpha = np.zeros(n, dtype=np.float64)
        self.b = 0.0
        if K_train is None:
            K_train = kernel(X, X, self.gamma)
        t=1
        for epoch in range(self.epochs):
            perm=rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                K_b_all = K_train[batch_idx, :]
                f_b = K_b_all @ (self.alpha * self.y_train) + self.b

                viol_mask = (yb * f_b) < 1.0
                eta_t = 1.0 / (self.lambda_reg * t)
                self.alpha *= (1.0 - eta_t * self.lambda_reg)

                if np.any(viol_mask):
                    viol_idx = batch_idx[viol_mask]
                    self.alpha[viol_idx] += eta_t
                    grad_b = - np.mean(yb[viol_mask]) 
                    self.b -= eta_t * grad_b
                t += 1
            # track hinge loss
            K_full = K_train
            f_all=K_full @ (self.alpha * self.y_train) + self.b
            hinge= np.maximum(0,1-self.y_train*f_all).mean()
            reg=0.5*self.lambda_reg*(self.alpha@(K_full@self.alpha))
            self.history_["hinge_loss"].append(float(hinge+reg))
            self
        return self

    def decision_function(self, X):
        K_test = kernel(X, self.X_train, self.gamma)
        return K_test @ (self.alpha * self.y_train) + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)