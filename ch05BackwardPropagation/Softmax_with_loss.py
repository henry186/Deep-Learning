from functions import softmax, cross_entropy_error


class SoftmaxWithLoss:
    def _init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        # Idk why it needs to divide by batch_size
        dx = (self.y - self.t) / batch_size

        return dx
