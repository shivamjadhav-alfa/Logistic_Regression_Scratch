#logistic regg
import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.w=None
        self.c=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.w=np.zeros(n_features)
        self.c=0
        for _ in range(self.n_iters):
           linear_model=np.dot(X,self.w)+self.c
           y_predicted=self._sigmoid(linear_model)
           dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
           dc=(1/n_samples)*np.sum(y_predicted-y)
           self.w -=self.lr*dw
           self.c -=self.lr*dc
    
    
    def predict(self,X):
        linear_model=np.dot(X,self.w)+self.c
        y_predicted=self._sigmoid(linear_model)
        y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))