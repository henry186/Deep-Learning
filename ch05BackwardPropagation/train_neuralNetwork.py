import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# load training data
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

network = TwoLayerNet(inputLayer=784, hiddenLayer=50, outputLayer=10)

# hyperparameters
iters_num = 70000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size, batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # backward propagating to calculate gradient
    grad = network.gradient(x_batch, t_batch)

    # learning the parameters
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.param[key] -= learning_rate * grad[key]
    # calculat loss
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # calculate accuracy every epoch(fully seeing the data)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)
        test_acc = network.accuracy(x_batch, t_batch)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
