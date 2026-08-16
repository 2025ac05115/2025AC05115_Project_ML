import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=5000):
        self.lr = lr
        self.n_iter = n_iter
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iter):
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
