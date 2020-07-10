import numpy as np
from NeuralNets import NeuralNetworkModel
import matplotlib.pyplot as plt

def randomWeights(r,c):

        epsilon = 0.12
        Theta = np.random.rand(r,c) * 2* epsilon - epsilon

        return Theta


#data = np.loadtxt('data', usecols=range(400), delimiter=',')
#np.save('data',data)

X = np.load('Input Data/Examples/data.npy') # loads input data matrix X
X = np.mat(X) # Convert into numpy matrix

y = np.load('Input Data/Labels/labels.npy') # loads the corresponded labels for X
y = np.mat(y).transpose() # convert into column vector

#Theta1 = np.loadtxt('Theta1', usecols=range(401), delimiter=',')
#Theta2 = np.loadtxt('Theta2', usecols=range(26), delimiter=',')


Theta = [randomWeights(25,401), randomWeights(10,26)]
NN = NeuralNetworkModel(3,X,y,Theta,0,10) # Creating Neural Net Model


Theta = NN.gradDescent(0.3,1500)
np.save('Theta1Learned',Theta[0])
np.save('Theta2Learned',Theta[1])

plt.plot(NN.Jhistory[1:]); plt.show()

predict = NN.predict(X)

for i in range(0,5000):

    predict[i] = predict[i].argmax()

print(np.mean(predict == y)*100)