# Momentum:  v = av - n(=lr) * grad(W)
# W = W + v

import numpy as np


class Momentum:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    # update the parameters of the model
    def update(self, params, grads):  # params, grads are dictionary type
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            for key in params.keys():
                self.v[key] = self.momentum*self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]
