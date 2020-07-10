import numpy as np

class NeuralNetworkModel:
    Jhistory = 0
    def __init__(self,a,X,y,Theta,Lambda,K):
        self.m = y.shape[0] # Number of examples
        self.n = X.shape[1] # Number of features
        self.a = np.zeros((a,1)) # Neurons
        self.z = a # Function of line
        self.Theta = Theta # Weights matrices
        self.K = K # Numver of labels
        self.Lambda = Lambda # Regularization factor
        self.X = np.append(np.ones((y.shape[0],1)),X,1)
        self.y = y
        self.Ylabel = range(0,K) == y
        #self.Ylabel = self.Ylabel[:,1:]


    def sigmoid(self,Z):
        Z = np.mat(Z)
        return 1/(1+np.exp(-Z))

    def sigmoidGradient(self,Z):
        Z = np.mat(Z)
        return np.multiply(self.sigmoid(Z),(1-self.sigmoid(Z)))

    def cost(self,X,Theta,Lambda):

        h = (self.feedForward(X,Theta))[2]
        RegularizationTerm = Lambda/(2*self.m) * (np.sum(np.sum(np.power(Theta[0],2))) + np.sum(np.sum(np.power(Theta[1],2))))
        J = 1/self.m * np.sum( np.sum( np.multiply( -1*self.Ylabel,np.log(h) ) - np.multiply( (1-self.Ylabel),np.log(1-h) ) ) ) + RegularizationTerm

        return J

    def feedForward(self,X,Theta):

        z2 = self.X@(Theta[0].transpose())
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones((self.m,1)),a2,1)
        z3 = a2@(Theta[1].transpose())
        a3 = self.sigmoid(z3)

        return [z2,a2,a3]


    def backPropagation(self):
        
        [z2,a2,a3] = self.feedForward(self.X,self.Theta)
        
        err3 = a3 - self.Ylabel
        err2 = np.multiply(err3*self.Theta[1][:,1:],self.sigmoidGradient(z2))
        
        delta1 = self.X.transpose()*err2
        delta2 = a2.transpose()*err3

        Grad1 = 1/self.m*delta1
        Grad2 = 1/self.m*delta2

        return [Grad1.transpose(),Grad2.transpose()]

    def gradDescent(self,alpha,iter):

        for i in range(0,iter):

            self.Theta[0] = self.Theta[0] - alpha*self.backPropagation()[0]
            self.Theta[1] = self.Theta[1] - alpha*self.backPropagation()[1]

            self.Jhistory = np.append(self.Jhistory,self.cost(self.X,self.Theta,self.Lambda))
            print(self.Jhistory[i])
        return [self.Theta[0],self.Theta[1]]

    def predict(self,X):

        return self.feedForward(X,self.Theta)[2]