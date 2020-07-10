import numpy as np
from NeuralNets import NeuralNetworkModel
import matplotlib.pyplot as plt

X = np.loadtxt('ex2data2.txt', usecols=range(2), delimiter=',') # loads input data matrix X
y = np.loadtxt('ex2data2.txt', usecols=range(2,3), delimiter=',') # loads the corresponded labels for X

# Plot input data
Class0 = np.where(y==0); Class1 = np.where(y==1)
plt.scatter(X[Class0,0],X[Class0,1],c="Red")
plt.scatter(X[Class1,0],X[Class1,1],c="Blue")
# Convert to numpy matrix
X = np.mat(X); y = np.mat(y).transpose()

Theta = [np.load('Theta1Learned.npy'),np.load('Theta2Learned.npy')]

NN = NeuralNetworkModel(3,X,y,Theta,0,2) # Creating Neural Net Model

#xgrid = np.linespace(-1,1.5,50)
#ygrid = np.linespace(-1,1.5,50)

#for i in range(0,xgrid.shape[0]):
   # for j in range(0,ygrid[0]):
       # H[i,j] =  
predict = NN.predict(X)

for i in range(0,118):

    predict[i] = predict[i].argmax()
    
print(np.mean(predict == y)*100)
