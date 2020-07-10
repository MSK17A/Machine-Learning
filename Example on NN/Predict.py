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

xgrid = np.linspace(-1,1.5,150)
ygrid = np.linspace(-1,1.5,150)

Grid = np.c_[xgrid,ygrid]
Z = np.zeros([xgrid.shape[0],ygrid.shape[0]])


for i in range(0,xgrid.shape[0]):
    for j in range(0,ygrid.shape[0]):
        X = np.mat([xgrid[i],ygrid[j]])
        Z[i,j] = NN.predict(X).argmax()

plt.contour(Grid[:,0],Grid[:,1],Z.transpose(),[0,0]); plt.show()


#for i in range(0,50):

#    predict[i] = predict[i].argmax()
    
#print(np.mean(predict == y)*100)
