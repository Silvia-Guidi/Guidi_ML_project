import numpy as np
## SVM 
class LinearSVM:
    def __init__(self, lambda_reg=1e-3, epochs=15, batch_size=64, shuffle=True, random_state=42, degree=None, coef0=1):
        self.lambda_reg = float(lambda_reg)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.w = None
        self.b = 0.0
        self.history_ = {"obj": []} 
        self.degree=degree
        self.coef0=coef0
        self.history_ = {"obj": [], "val_obj": [], "train_acc": [], "val_acc": []} 
    
    def _hingelosses(self, X, y):
        margins = 1 - y * (X @ self.w + self.b)
        hinge = np.maximum(0.0, margins).mean()
        reg = 0.5 * self.lambda_reg * (self.w @ self.w)
        return reg + hinge
    
    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state) # random choice of the element
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)  #init the weight and bias vectors
        self.b = 0.0
        t = 0
        idx = np.arange(n)
        for epoch in range(self.epochs):  
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, n, self.batch_size):  #update epoch and reg term
                t += 1
                eta = 1.0 / (self.lambda_reg * t)
                batch = idx[start:start + self.batch_size]  #get the example
                Xb = X[batch]
                yb = y[batch]
                # find violators: y*(w·x + b) < 1
                margins = yb * (Xb @ self.w + self.b)
                viol = margins < 1.0
                m = max(1, viol.sum())  # avoid divide-by-zero
                # shrink w for regularization
                self.w *= (1.0 - eta * self.lambda_reg)
                # subgradient step from violators
                if viol.any():
                    Xv = Xb[viol]
                    yv = yb[viol]
                    self.w += (eta / m) * (Xv.T @ yv)
                    # bias gets NO regularization
                    self.b += (eta / m) * yv.sum()
            # update history
            self.history_["obj"].append(self._hingelosses(X, y))
            train_acc = (self.predict(X) == y).mean()
            self.history_["train_acc"].append(train_acc)
            if X_val is not None and y_val is not None:
                val_obj = self._hingelosses(X_val, y_val)
                val_acc = (self.predict(X_val) == y_val).mean()
                self.history_["val_obj"].append(val_obj)
                self.history_["val_acc"].append(val_acc)

        return self
    
    def decision_function(self, X):
        return X @ self.w + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
    def score(self, X, y):
        return (self.predict(X) == y).mean()

# Logistic Regression
class LogisticRegression:
    def __init__(self, lambda_reg=1e-3, epochs=50, batch_size=64, eta=0.1,
                 shuffle=True, random_state=42, degree=None, coef0=1):
        self.lambda_reg = float(lambda_reg)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.eta = float(eta)  # learning rate
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)
        self.w = None
        self.b = 0.0
        self.history_ = {"loss": []}  # track loss per epoch
        self.degree=degree
        self.coef0=coef0
        self.history_ = {"loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def _loss(self, X, y):
        z = X @ self.w + self.b
        # logistic loss
        log_loss = np.log(1 + np.exp(-y * z)).mean()
        # L2 regularization
        reg = 0.5 * self.lambda_reg * (self.w @ self.w)
        return log_loss + reg
    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        idx = np.arange(n)
        for epoch in range(self.epochs):
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb = X[batch]
                yb = y[batch]
                z = Xb @ self.w + self.b
                p = self._sigmoid(-yb * z)  # σ(-y*(w·x+b))
                # gradient computation
                grad_w = -(Xb.T @ (yb * p)) / len(yb) + self.lambda_reg * self.w
                grad_b = -(yb * p).sum() / len(yb)
                # gradient descent update
                self.w -= self.eta * grad_w
                self.b -= self.eta * grad_b
            # update history
            self.history_["loss"].append(self._loss(X, y))
            train_acc = (self.predict(X) == y).mean()
            self.history_["train_acc"].append(train_acc)
            if X_val is not None and y_val is not None:
                val_loss = self._loss(X_val, y_val)
                val_acc = (self.predict(X_val) == y_val).mean()
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)
        return self

    def decision_function(self, X):
        return X @ self.w + self.b
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
    def score(self, X, y):
        return (self.predict(X) == y).mean()