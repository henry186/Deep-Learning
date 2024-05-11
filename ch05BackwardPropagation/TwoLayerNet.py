import numpy as np
from layers import *
from functions import *
from gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, inputLayer, hiddenLayer, outputLayer, weight_init_std=0.01):
        # initialize the parameters
        self.param = {}
        self.param['W1'] = weight_init_std * \
            np.random.randn(inputLayer, hiddenLayer)
        self.param['b1'] = weight_init_std * np.zeros(hiddenLayer)
        self.param['W2'] = weight_init_std * \
            np.random.randn(hiddenLayer, outputLayer)
        self.param['b2'] = weight_init_std * np.zeros(outputLayer)

        # implement the layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.param['W1'], self.param['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.param['W2'], self.param['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.param['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.param['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.param['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.param['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['W2'] = self.layers['Affine2'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['b2'] = self.layers['Affine2'].db

        return grads
