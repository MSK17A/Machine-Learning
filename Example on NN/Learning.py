import numpy as np
from NeuralNets import NeuralNetworkModel
import matplotlib.pyplot as plt

def randomWeights(r,c):

        epsilon = 0.12
        Theta = np.random.rand(r,c) * 2* epsilon - epsilon

        return Theta


#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.loadtxt('ex2data2.txt', usecols=range(2), delimiter=',') # loads input data matrix X
 # Convert into numpy matrix

y = np.loadtxt('ex2data2.txt', usecols=range(2,3), delimiter=',') # loads the corresponded labels for X
#y = np.mat(y).transpose() # convert into column vector

Class0 = np.where(y==0); Class1 = np.where(y==1)
plt.scatter(X[Class0,0],X[Class0,1],c="Red")
plt.scatter(X[Class1,0],X[Class1,1],c="Blue"); plt.show()

X = np.mat(X); y = np.mat(y).transpose()

Theta = [randomWeights(4,3), randomWeights(2,5)]
NN = NeuralNetworkModel(3,X,y,Theta,0,2) # Creating Neural Net Model


Theta = NN.gradDescent(0.3,9000)
np.save('Theta1Learned',Theta[0])
np.save('Theta2Learned',Theta[1])

plt.plot(NN.Jhistory[1:]); plt.legend(['Loss Function']); plt.show()

predict = NN.predict(X)

for i in range(0,118):

    predict[i] = predict[i].argmax()

print(np.mean(predict == y)*100)