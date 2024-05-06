import numpy as np

def cross_entropy_error(y,t): #y是算出的結果 t是testing data的label
    delta = 1e-7
    return -np.dot(t, np.log(y + delta))

t = np.array([0,0,1,0,0,0,0,0,0,0]) #answer = 2
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05,0.1, 0.0, 0.1, 0.0, 0.0])

e = cross_entropy_error(y,t)
print(e)
