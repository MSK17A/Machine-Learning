import numpy as np
from NeuralNets import NeuralNetworkModel
import matplotlib.pyplot as plt

#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.load('Input Data/Examples/data.npy') # loads input data matrix X
X = np.mat(X) # Convert into numpy matrix

y = np.load('Input Data/Labels/labels.npy') # loads the corresponded labels for X
y = np.mat(y).transpose() # convert into column vector

#Theta1 = np.loadtxt('Theta1', usecols=range(401), delimiter=',')
#Theta2 = np.loadtxt('Theta2', usecols=range(26), delimiter=',')
Theta = [np.load('Theta1Learned.npy'),np.load('Theta2Learned.npy')]

NN = NeuralNetworkModel(3,X,y,Theta,0,10) # Creating Neural Net Model

predict = NN.predict(X)

for i in range(0,5000):

    predict[i] = predict[i].argmax()
    
print(np.mean(predict == y)*100)