import numpy as np
import NueralNets as nn

#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.load('Nueral Network/data.npy') # loads input data matrix X
X = np.mat(X) # Convert into numpy matrix

y = np.load('Nueral Network/labels.npy') # loads the corresponded labels for X
y = np.mat(y).transpose() # convert into column vector

#Theta1 = np.loadtxt('Nueral Network/Theta1', usecols=range(401), delimiter=',')
#Theta2 = np.loadtxt('Nueral Network/Theta2', usecols=range(26), delimiter=',')
Theta = [np.load('Nueral Network/Theta1.npy'),np.load('Nueral Network/Theta2.npy')]

print(nn.cost(X,y,Theta,1,10))