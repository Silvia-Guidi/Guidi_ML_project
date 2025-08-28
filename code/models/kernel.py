import numpy as np

# Kernel
def kernel(X, Y, gamma=0.1):
    # compute squared Euclidean distance between each pair
    X_norm = np.sum(X**2, axis=1)[:, np.newaxis]  # (n_X, 1)
    Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]  # (1, n_Y)
    sq_dists = X_norm + Y_norm - 2*np.dot(X, Y.T)
    
    K = np.exp(-gamma * sq_dists)
    return K

def predict_kernel(X_test, X_train, alpha, b, gamma):
    K_test = kernel(X_test, X_train, gamma)
    return np.sign(K_test @ alpha + b)

class KernelLogisticRegression:
    def __init__(self, lambda_reg=0.1, gamma=0.1, epochs=50, eta=0.1):
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.epochs = epochs
        self.eta = eta
        self.alpha = None
        self.b = 0.0
        self.X_train = None  # store training features

    def _stable_sigmoid(self, z):
        # Numerically stable sigmoid
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def fit(self, X, y):
        n = X.shape[0]
        self.X_train = X
        K_train = kernel(X, X, self.gamma)  # kernel of training data
        self.alpha = np.zeros(n)
        self.b = 0.0

        for epoch in range(self.epochs):
            f = K_train @ self.alpha + self.b
            p = self._stable_sigmoid(-y * f)  # stable sigmoid
            grad_alpha = (K_train @ (-y * p)) / n + self.lambda_reg * (K_train @ self.alpha)
            grad_b = (-y * p).mean()
            self.alpha -= self.eta * grad_alpha
            self.b -= self.eta * grad_b
        return self

    def predict(self, X):
        K_test = kernel(X, self.X_train, self.gamma)
        f = K_test @ self.alpha + self.b
        return np.where(f >= 0, 1, -1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

class KernelSVM:
    def __init__(self, lambda_reg=0.01, gamma=0.1, epochs=15, eta=0.1):
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.epochs = epochs
        self.eta = eta
        self.alpha = None
        self.b = 0.0
        self.K_train = None
        self.y_train = None
        self.history_ = {"hinge_loss": []}

    def fit(self, X, y):
        n = X.shape[0]
        self.X_train = X  
        self.K_train = kernel(X, X, self.gamma)
        self.alpha = np.zeros(n)
        self.b = 0.0
        self.y_train = y.copy()
        self.history_ = {"hinge_loss": []}

        for epoch in range(self.epochs):
            # compute decision values
            f = self.K_train @ (self.alpha * y) + self.b
            margins = y * f

            # hinge loss
            hinge_loss = np.maximum(0, 1 - margins).mean() + 0.5 * self.lambda_reg * (self.alpha @ self.K_train @ self.alpha)
            self.history_["hinge_loss"].append(hinge_loss)

            # subgradient
            viol = margins < 1
            grad_alpha = np.zeros(n)
            grad_alpha[viol] = 1.0 / n  # simple average over violators
            grad_alpha = self.K_train @ (grad_alpha * y) + self.lambda_reg * (self.K_train @ self.alpha)
            grad_b = (y[viol].sum()) / n

            # update
            self.alpha -= self.eta * grad_alpha
            self.b -= self.eta * grad_b
        return self

    def decision_function(self, X):
        K_test = kernel(X, self.X_train, self.gamma)  
        return K_test @ (self.alpha * self.y_train) + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
    def score(self, X, y):
        return (self.predict(X) == y).mean()