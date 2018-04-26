import numpy as np
import pandas as pd
from numpy import dot

#read in the dataset
dataset = pd.read_csv('./data/data.csv')
temp = dataset.iloc[:, 2:5]
temp['x0'] = 1
X = temp.iloc[:, [3, 0, 1, 2]]
Y = dataset.iloc[:, 1].values.reshape(150, 1)


#Gradient Descent
#initial parameters and learning rate
theta = (np.random.random([1,4])*2-1).reshape(4, 1)
alpha = 0.1
temp = theta
X0 = X.iloc[:, 0].values.reshape(150, 1)
X1 = X.iloc[:, 1].values.reshape(150, 1)
X2 = X.iloc[:, 2].values.reshape(150, 1)
X3 = X.iloc[:, 3].values.reshape(150, 1)
for i in range(10000):
    temp[0] = theta[0] + alpha*np.sum((Y-dot(X, theta))*X0)/150.
    temp[1] = theta[1] + alpha*np.sum((Y-dot(X, theta))*X1)/150.
    temp[2] = theta[2] + alpha*np.sum((Y-dot(X, theta))*X2)/150.
    temp[3] = theta[3] + alpha*np.sum((Y-dot(X, theta))*X3)/150.
    theta = temp
print(theta)


#Stochastic Gradient Descent
theta = (np.random.random([1,4])*2-1).reshape(4, 1)
alpha = 0.1
X = X.values.reshape(150, 4)
for m in range(15000):
    global_error = 0
    for n in range(len(dataset)):
        temp = theta + alpha * (1/150.)* np.reshape((Y[n] - dot(X[n, :], theta)) * X[n, :],[4,1])
        theta = temp
        global_error = np.sum(np.reshape( Y - dot(X,theta),[len(dataset),1]))/len(dataset)
        #print(global_error)
    if abs(global_error) <= 0.001:
        break
print(theta)




