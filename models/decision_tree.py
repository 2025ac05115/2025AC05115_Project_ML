import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
    
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts/len(y))**2)
    
    def split(self, X, y, feature, threshold):
        left_idx = X[:,feature] <= threshold
        right_idx = X[:,feature] > threshold
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]
    
    def best_split(self, X, y):
        best_feat, best_thresh, best_score = None, None, 1e9
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:,feat])
            for t in thresholds:
                X_left, y_left, X_right, y_right = self.split(X,y,feat,t)
                if len(y_left)==0 or len(y_right)==0: continue
                score = (len(y_left)*self.gini(y_left) + len(y_right)*self.gini(y_right)) / len(y)
                if score < best_score:
                    best_feat, best_thresh, best_score = feat, t, score
        return best_feat, best_thresh
    
    def build(self, X, y, depth):
        if depth==self.max_depth or len(np.unique(y))==1:
            return int(np.argmax(np.bincount(y)))
        feat, thresh = self.best_split(X,y)
        if feat is None: return int(np.argmax(np.bincount(y)))
        X_left,y_left,X_right,y_right = self.split(X,y,feat,thresh)
        return {"feat":feat,"thresh":thresh,
                "left":self.build(X_left,y_left,depth+1),
                "right":self.build(X_right,y_right,depth+1)}
    
    def fit(self,X,y):
        self.tree = self.build(X,y,0)
    
    def predict_one(self,x,node):
        if isinstance(node,int): return node
        if x[node["feat"]] <= node["thresh"]:
            return self.predict_one(x,node["left"])
        else:
            return self.predict_one(x,node["right"])
    
    def predict(self,X):
        return np.array([self.predict_one(x,self.tree) for x in X])
