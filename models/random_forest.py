import numpy as np
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self,n_estimators=10,max_depth=3):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
    
    def fit(self,X,y):
        self.trees=[]
        n=len(X)
        for _ in range(self.n_estimators):
            idx=np.random.choice(n,n,replace=True)
            X_s,y_s=X[idx],y[idx]
            tree=DecisionTree(max_depth=self.max_depth)
            tree.fit(X_s,y_s)
            self.trees.append(tree)
    
    def predict(self,X):
        preds=np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(preds,axis=0)).astype(int)
