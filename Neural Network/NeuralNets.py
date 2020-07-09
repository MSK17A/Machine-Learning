import numpy as np


def sigmoid(Z):
    Z = np.mat(Z)
    return 1/(1+np.exp(-Z))

def cost(X,y,Theta,Lambda,K):
    m = y.shape[0]
    Y = range(1,K+1) == y

    h = feedForward(X,y,Theta)
    RegularizationTerm = Lambda/(2*m) * (np.sum(np.sum(np.power(Theta[0],2))) + np.sum(np.sum(np.power(Theta[1],2))))
    J = 1/m * np.sum( np.sum( np.multiply( -1*Y,np.log(h) ) - np.multiply( (1-Y),np.log(1-h) ) ) ) + RegularizationTerm

    return J

def feedForward(X,y,Theta):

    X = np.append(np.ones((y.shape[0],1)),X,1)

    z2 = X@(Theta[0].transpose())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((y.shape[0],1)),a2,1)
    z3 = a2@(Theta[1].transpose())
    a3 = sigmoid(z3)

    return a3

def randomWeights(r,c):

    epsilon = 0.12
    Theta = np.random.rand(r,c) * 2* epsilon - epsilon

    return Theta

#def backPropagation(X,y,Theta):