import numpy as np
from common.functions import *
from common.util import im2col, col2im
from common.matmul_quire import *


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        _t = type(x.flat[0])

        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = _t(0)

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Flatten:
    def __init__(self):
        self.original_x_shape = None

    def forward(self, x):
        # Reshape into NWHC format
        x = x.transpose(0, 2, 3, 1)
        # # The corresponding tensor
        self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        # Restore the shape of the input data (corresponding tensor)
        dout = dout.reshape(*self.original_x_shape)
        # reshape into NCHW format
        dout = dout.transpose(0, 3, 1, 2)
        return dout.reshape(original_x_shape)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        # self.original_x_shape = None
        # Derivatives of weight and bias parameters
        self.dW = None
        self.db = None

    def forward(self, x):
        # # The corresponding tensor
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        self.x = x

        # out = np.dot(self.x, self.W) + self.b
        # print(type(self.x.item(0)), self.x.shape)
        # print(type(self.W.item(0)), self.W.shape)
        # print(type(self.b.item(0)))
        out = matmul_quire(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        # dx = np.dot(dout, self.W.T)
        # self.dW = np.dot(self.x.T, dout)
        dx = matmul_quire(dout, self.W.T)
        self.dW = matmul_quire(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # Restore the shape of the input data (corresponding tensor)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax output
        self.t = None  # Monitoring data

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # Supervising the case where the data is one-hot-vector
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # Intermediate data (used for backward)
        self.x = None
        self.col = None
        self.col_W = None

        # Gradients of weights and bias parameters
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        # FH, FW, C, FN  = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        # print("Conv Forward!")
        # print(type(x.item(0)), x.shape)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        # col_W = self.W.reshape(-1, FN)

        # print(type(col.item(0)), col.shape)
        # print(type(col_W.item(0)), col_W.shape)

        # out = np.dot(col, col_W) + self.b
        out = matmul_quire(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # print(type(out.item(0)), out.shape)

        self.x = x
        self.col = col
        self.col_W = col_W

        # print(self.W)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        # self.dW = np.dot(self.col.T, dout)
        self.dW = matmul_quire(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dcol = np.dot(dout, self.col_W.T)
        dcol = matmul_quire(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx
