import numpy as np

class LinearModel:
    theta = 0
    J_log = 0
    def __init__(self, theta, J_log):
        self.theta = theta
        self.J_log = J_log

    @classmethod
    def Cost(cls,X,y,theta):
        "X = input \n y = output \n theta = weights"
        m = np.size(y)
        # h = X@theta; h = np.transpose(h)
        # J = 1/(2*m)*(h-y)@ np.transpose((h-y))
        h = X@theta
        J = 1/(2*m) * np.sum(np.power((h-y),2))

        return J

    @classmethod
    def gradDescent(cls,X,y,theta,alpha,iter):

        m = np.size(y)
        J_history = np.zeros((iter,1))
        cls.theta = theta
        for i in range(0,iter):
            dJ = (alpha/m) * (X.transpose() @ (X@(cls.theta)-y))
            cls.theta = cls.theta - dJ # theta = theta - alpha/m * X'*(X*theta-y);
            J_history[i] = cls.Cost(X, y, cls.theta)
            
        cls.J_log = J_history
        return cls.theta,cls.J_log

    @classmethod
    def FeatureNormalization(cls,X):

        mu = np.mean(X)
        sigma = np.std(X)

        X_norm = (X-mu)/sigma
        
        return X_norm,sigma,mu

    @classmethod
    def predict(cls,X):
        return X@cls.theta