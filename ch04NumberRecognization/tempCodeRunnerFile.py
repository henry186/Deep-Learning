from mnist import load_mnist
from TwoLayerNet import TwoLayerNet
import numpy as np

def get_data():
    ((x_train, t_train), (x_test, t_test)) = load_mnist(normalize=True, one_hot_label=True) #flatten default -> True
    return x_train, t_train


# get the training data
x,t = get_data()
print(x.shape)
print(t.shape)

print(str(len(x)))
train_size = x.shape[0]     #60000
batch_size = 100
batch_mask = np.random.choice(train_size, batch_size) #from 0~60000 隨機取10個數
x_batch = x[batch_mask]
t_batch = t[batch_mask]
print(x_batch.shape)
print(t_batch.shape)