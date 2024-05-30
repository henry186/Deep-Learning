# AdaGrad : h = h + grad(W) 。 grad(W) (sum of (grad( (ele-wise-square(W) )) )
# W = W - n(=lr) * ( 1/sqrt(h) ) * grad(W)
# dynamic learning rate can avoid big parameters continuously growing bigger
import numpy as np


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
        for key, val in params.items():
            self.h[key] = np.zero_like(val)
        for key in params.key():
            self.h[key] += grads[key]**2
            # Add 1e-7 to the denominator to avoid division by zero
            params[key] -= self.lr * (grads[key] / np.sqrt(self.h[key]) + 1e-7)
