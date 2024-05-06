from mnist import load_mnist
from TwoLayerNet import TwoLayerNet
import numpy as np
import matplotlib.pyplot as plt

# get the training data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) #flatten default -> True

train_loss_list = [] # store the loss after the training every time

# set the hyperparameters
iters_nums = 1000                  # learning times
train_size = x_train.shape[0]       # dataset size = 60000
batch_size = 100                    # 100 inputs per learn
learning_rate = 0.1                 # move distance after learning every time

for i in range(iters_nums):
    # select a number(batch size) of data
    batch_mask = np.random.choice(train_size, batch_size) #from 0~60000 隨機取10個數
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # create a object of TwoLayerNet with 50 hidden neurals
    network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

    # calculate the gradient
    grad = network.numerical_gradient(x_batch,t_batch)

    # update the parameters
    network.params['W1'] -= learning_rate * grad['W1']
    network.params['b1'] -= learning_rate * grad['b1']
    network.params['W2'] -= learning_rate * grad['W2']
    network.params['b2'] -= learning_rate * grad['b2']

    # record the loss
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

# Write the parameters to a text file
with open('parameters.txt', 'w') as f:
    for key, value in network.params.items():
        f.write(f'{key}:\n{value}\n\n')

# 繪製圖表
plt.plot(range(iters_nums), train_loss_list, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.show()

# print(train_loss_list)