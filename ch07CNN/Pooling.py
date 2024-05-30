from common.util import im2col
import numpy as np


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # N: num of input image, C: num of channels(ex:an RGB image has 3 Channels), H:image Height, W:image width
        N, C, H, W = x.shape

        # output size:(OH,OW) OH = (H + 2P - FH) / S + 1
        out_H = (H + 2*self.pad - self.pool_h) / self.stride + 1
        out_W = (W + 2*self.pad - self.pool_w) / self.stride + 1

        # unfold the 4D-input to 2D
        # col shape = (OH * OW, C * poll_h * pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # keep only the max value
        out = np.max(out, axis=1)

        # reshape to 4D (FN,C,OH,OW)
        # transpose because the original out.reshape will produce(FN,OH,OW,C)
        out = out.reshape(N, out_H, out_W, C).transpose(0, 3, 1, 2)

        return out
