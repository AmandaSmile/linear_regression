import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import dot

#read in the dataset and divide it
dataset = pd.read_csv('./data/data.csv')
l = int(len(dataset)*0.8)
train = dataset.iloc[0:l,:]
test = dataset.iloc[l+1 : ,:]

# random initialize parameters , learning rate and threshold, threshold is used to judgewhether to continue train
theta = np.random.random([1,4])*2-1
alpha = 0.001
threshold = 0.01
X = np.ones([len(train),4])
X[:,1:4] = train.values[:,2:5]
Y = train.values[:,1]

#Training
for i in range(100):
    for j in range(len(train)):
        error = dot(theta, X[i, :].T) - Y[i]
        theta = theta - alpha * error * X[i, :]
    global_error = sum(dot(X, theta.T) - np.reshape(Y,[X.shape[0],1])) / X.shape[0]
    #print(global_error)
    print(theta)
    if abs(global_error)<=0.01:
        break

#print the parameters
print(theta)

#Testing
print('---------------------------')
print('Testing')
X_test = np.ones([len(test),4])
X_test[:,1:4] = test.values[:,2:5]
Y_test = test.values[:,1]
Test_error = sum(dot(X_test, theta.T) - np.reshape(Y_test,[X_test.shape[0],1])) / X_test.shape[0]
print(Test_error)