import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearModel as LRM


data = np.loadtxt('ex1data1.txt', usecols=range(2), delimiter=',')
X = data[:,range(1)]; y = data[:,range(1,2)]
X = np.mat(X); y = np.mat(y)
#[X,sigma,mu] = LRM.FeatureNormalization(X)
#X = np.c_[oneVec, np.array(X)]

m = y.shape[0]
X = np.append(np.ones((m,1)), X, axis=1)
n = X.shape[1]

theta = np.zeros((n,1))

LRM.gradDescent(X,y,theta,0.01,1500)
plt.plot(LRM.J_log); plt.show()
plt.scatter(data[:,range(1)],data[:,range(1,2)]); plt.plot(data[:,range(1)],X@(LRM.theta)); plt.show()

print(LRM.predict([1,3.5])*10000)