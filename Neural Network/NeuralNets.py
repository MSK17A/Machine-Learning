import numpy as np

class NeuralNetworkModel:

    a = 0
    z = 0
    m = 0
    n = 0
    def __init__(self,a,X,y):
        self.m = y.shape[0] # Number of examples
        self.n = X.shape[1] # Number of features
        self.a = np.zeros((a,1))
        self.z = a


    def sigmoid(self,Z):
        Z = np.mat(Z)
        return 1/(1+np.exp(-Z))

    def cost(self,X,y,Theta,Lambda,K):
        Y = range(1,K+1) == y

        h = self.feedForward(X,y,Theta)
        RegularizationTerm = Lambda/(2*self.m) * (np.sum(np.sum(np.power(Theta[0],2))) + np.sum(np.sum(np.power(Theta[1],2))))
        J = 1/self.m * np.sum( np.sum( np.multiply( -1*Y,np.log(h) ) - np.multiply( (1-Y),np.log(1-h) ) ) ) + RegularizationTerm

        return J

    def feedForward(self,X,y,Theta):

        X = np.append(np.ones((y.shape[0],1)),X,1)

        z2 = X@(Theta[0].transpose())
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones((y.shape[0],1)),a2,1)
        z3 = a2@(Theta[1].transpose())
        a3 = self.sigmoid(z3)

        return a3

    def randomWeights(self,r,c):

        epsilon = 0.12
        Theta = np.random.rand(r,c) * 2* epsilon - epsilon

        return Theta

    #def backPropagation(y,Theta):