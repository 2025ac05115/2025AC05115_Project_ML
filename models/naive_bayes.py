import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        for c in self.classes:
            X_c = X[y==c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.prior[c] = len(X_c) / len(X)
    
    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.prior[c])
                likelihood = -0.5*np.sum(np.log(2*np.pi*self.var[c]))
                likelihood -= 0.5*np.sum(((x-self.mean[c])**2)/self.var[c])
                posteriors.append(prior+likelihood)
            preds.append(np.argmax(posteriors))
        return np.array(preds)
