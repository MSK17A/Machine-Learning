import numpy as np
from NeuralNets import NeuralNetworkModel

#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.load('data.npy') # loads input data matrix X
X = np.mat(X) # Convert into numpy matrix

y = np.load('labels.npy') # loads the corresponded labels for X
y = np.mat(y).transpose() # convert into column vector

#Theta1 = np.loadtxt('Theta1', usecols=range(401), delimiter=',')
#Theta2 = np.loadtxt('Theta2', usecols=range(26), delimiter=',')
Theta = [np.load('Theta1.npy'),np.load('Theta2.npy')]

NN = NeuralNetworkModel(3,X,y) # Creating Neural Net Model

print(NN.cost(X,y,Theta,1,10))