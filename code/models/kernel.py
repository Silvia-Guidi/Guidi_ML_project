import numpy as np

# Kernel
def kernel(X, Y, gamma=0.1):
    X_norm = np.sum(X**2, axis=1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]
    sq_dists = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-np.clip(sq_dists, 0, 1e6) / (2 * gamma))

def predict_kernel(X_test, X_train, alpha, b, gamma):
    K_test = kernel(X_test, X_train, gamma)
    return np.sign(K_test @ alpha + b)

class KernelLogisticRegression:
    def __init__(self, lambda_reg=0.1, gamma=0.1, epochs=100, eta=0.01, batch_size=None):
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.alpha = None
        self.b = 0.0
        self.X_train = None

    def _stable_sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n = X.shape[0]
        self.X_train = X.copy()
        y01 = (y + 1) / 2  # Map {-1,1} -> {0,1}
        K_train = kernel(X, X, self.gamma) + 1e-12 * np.eye(n)
        self.alpha = np.zeros(n)
        self.b = 0.0
        batch_size = self.batch_size or n  # full-batch if None
        for epoch in range(self.epochs):
            idx = np.random.choice(n, batch_size, replace=False)
            f_batch = K_train[idx] @ self.alpha + self.b
            p_batch = self._stable_sigmoid(f_batch)
            grad_alpha = K_train[:, idx] @ (p_batch - y01[idx]) / batch_size
            grad_alpha += self.lambda_reg * (K_train @ self.alpha)
            grad_b = (p_batch - y01[idx]).mean()
            self.alpha -= self.eta * grad_alpha
            self.b -= self.eta * grad_b

        return self

    def decision_function(self, X):
        K_test = kernel(X, self.X_train, self.gamma)
        f = K_test @ self.alpha + self.b
        return np.clip(f, -1e6, 1e6)

    def predict(self, X):
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

def poly_kernel(X, Y, degree=3, coef0=1):
    return (X @ Y.T + coef0) ** degree

class KernelSVM:
    def __init__(self, lambda_reg=0.01, degree=3, coef0=1, epochs=100, eta=0.01, batch_size=None):
        self.lambda_reg = lambda_reg
        self.degree = degree
        self.coef0 = coef0
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.history_ = {"hinge_loss": []}

    def fit(self, X, y):
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()
        K_train = poly_kernel(X, X, self.degree, self.coef0) + 1e-12 * np.eye(n)
        self.alpha = np.zeros(n)
        self.b = 0.0
        ay = self.alpha * y
        batch_size = self.batch_size or n

        for epoch in range(self.epochs):
            idx = np.random.choice(n, batch_size, replace=False)
            f = K_train[idx] @ ay + self.b
            margins = y[idx] * f
            hinge_loss = np.maximum(0, 1 - margins).mean() + 0.5 * self.lambda_reg * (self.alpha @ K_train @ self.alpha)
            self.history_["hinge_loss"].append(hinge_loss)

            viol = margins < 1
            grad_alpha = np.zeros(batch_size)
            grad_alpha[viol] = -y[idx[viol]] / batch_size
            grad_alpha_full = K_train[:, idx] @ grad_alpha + self.lambda_reg * (K_train @ self.alpha)
            grad_b = (-y[idx[viol]]).sum() / batch_size

            self.alpha -= self.eta * grad_alpha_full
            self.b -= self.eta * grad_b
            ay = self.alpha * y

        return self

    def decision_function(self, X):
        K_test = poly_kernel(X, self.X_train, self.degree, self.coef0)
        f = K_test @ (self.alpha * self.y_train) + self.b
        return np.clip(f, -1e6, 1e6)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return (self.predict(X) == y).mean()