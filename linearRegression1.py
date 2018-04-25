import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import dot

dataset = pd.read_csv('./data/data.csv')
#划分训练集和测试集
divid = int(len(dataset)*0.8)
train = dataset.iloc[0:divid,:]
test = dataset.iloc[divid + 1:,:]

X = np.ones([len(train),4])
#第一个值均为1,用作bias
X[:,1:4] = train.values[:,2:5]
Y = train.values[:,1]

#转化成矩阵
X = np.mat(X)
Y = np.mat(Y)

theta = dot(dot(inv(dot(X.T,X)),X.T),Y.T)
print(theta)

print(train.values[119,2:5])
print('True:',train.values[119,1])
print(dot(np.mat(theta[1:4]).reshape(1,3),train.values[119,2:5].T)+theta[0])

#Testing
print('---------------------------')
print('Testing')
X_test = np.ones([len(test),4])
X_test[:,1:4] = test.values[:,2:5]
Y_test = test.values[:,1]
Test_error = sum(dot(X_test, theta) - np.reshape(Y_test,[X_test.shape[0],1])) / X_test.shape[0]
print(Test_error)



