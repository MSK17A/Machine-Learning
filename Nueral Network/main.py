import numpy as np
import NueralNets as nn

#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.load('data.npy') # loads input data matrix X
X = np.mat(X) # Convert into numpy matrix

y = np.load('labels.npy') # loads the corresponded labels for X
y = np.mat(y).transpose() # convert into column vector

Theta1 = np.loadtxt('Theta1', usecols=range(401), delimiter=',')
Theta2 = np.loadtxt('Theta2', usecols=range(26), delimiter=',')
Theta = [Theta1,Theta2]

print(nn.cost(X,y,Theta,1,10))