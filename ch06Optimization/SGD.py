
# SGD: W = W - n(=lr) * gradient(W)
class SGD:

    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    # update the parameters of the model
    def update(self, params, grads):  # params, grads are dictionary type
        for key in params.key():
            params[key] -= self.lr * grads[key]
