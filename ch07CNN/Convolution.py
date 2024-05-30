from common.util import im2col
import numpy as np


class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # FN: num of filters, C:num of channels, FH: filter height, FW : filter width
        FN, C, FH, FW = self.W.shape
        # N: batch size(num of input feature maps), C: num of channels, H:height , W: width
        N, C, H, W = self.x.shape

        # output size:(OH,OW) OH = (H + 2P - FH) / S + 1
        out_h = int((H + 2 * self.pad - FH) / self.stride) + 1
        out_w = int((W + 2 * self.pad - FW) / self.stride) + 1

        # convert input shape from 4D to 2D
        col = im2col(x, FH, FW, self.stride, self.pad)

        # convert filter shape from 4D to 2D
        col_w = self.w.reshape(FN, -1).T

        # calculate output by matrix operations
        out = np.dot(col, col_w) + self.b

        # convert output shape from 2D to 4D
        out = out.reshape(FN, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
