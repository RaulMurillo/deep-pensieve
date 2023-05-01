import numpy as np
import softposit as sp
from tensorflow.keras.datasets import mnist
import sys
import os
import pickle
import csv
import time
from collections import OrderedDict
from common.layers import *
from common.matmul_quire import *
from common.read_weights import *
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

# Remove this for further evaluation
np.random.seed(1)


class SimpleConvNet:
    """Simple ConvNet
     affine-relu-affine-relu-affine-relu-affine--softmax

     Parameters
     ----------
     input_size: input size (28*28=784 in the case of MNIST)
     hidden_size_list: list of the number of neurons in the hidden layer (e.g. [100, 100, 100])
     output_size: output size (10 for MNIST)
     activation: 'relu' or 'sigmoid'
     weight_init_std: standard deviation of the specified weight (e.g. 0.01)
         When "relu" or "he" is specified, set "Initial value of He"
         When "sigmoid" or "xavier" is specified, "the initial value of Xavier" is set
     """

    def __init__(self, empty=True, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30,
                             'filter_size': 5, 'pad': 0, 'stride': 1},
                 output_size=10, weight_init_std=0.01, _t=sp.posit8):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # Initialization weight
        self.params = {}
        if empty:
            self.params['W1'] = np.empty(
                (6, 1, filter_size, filter_size), dtype=_t)
            self.params['b1'] = np.empty((6), dtype=_t)
            self.params['W2'] = np.empty(
                (16, 6, filter_size, filter_size), dtype=_t)
            self.params['b2'] = np.empty((16), dtype=_t)
            self.params['W3'] = np.empty((400, 120), dtype=_t)
            self.params['b3'] = np.empty((120), dtype=_t)
            self.params['W4'] = np.empty((120, 84), dtype=_t)
            self.params['b4'] = np.empty((84), dtype=_t)
            self.params['W5'] = np.empty((84, output_size), dtype=_t)
            self.params['b5'] = np.empty((output_size), dtype=_t)
        else:
            # TO-DO
            self.params['W1'] = weight_init_std * \
                np.random.randn(28*28, 1024)
            self.params['b1'] = np.zeros(1024)
            self.params['W2'] = weight_init_std * \
                np.random.randn(1024, 256)
            self.params['b2'] = np.zeros(256)
            self.params['W3'] = weight_init_std * \
                np.random.randn(256, 128)
            self.params['b3'] = np.zeros(128)
            self.params['W4'] = weight_init_std * \
                np.random.randn(128, output_size)
            self.params['b4'] = np.zeros(output_size)

        # Generating layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           filter_stride, filter_pad)
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Flatten'] = Flatten()

        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        i = 0
        for layer in self.layers.values():
            i += 1
            # print("$$$", layer, "$$$")
            # a = layer.W
            # print(x.shape)
            # print(x)
            # exit(1)
            x = layer.forward(x)
            # if (i == 6):
            #     print("Forward!", layer)
            #     # a = layer.W
            #     # print(a.shape)
            #     # print(a)
            #     x = x.transpose(0, 2, 3, 1)
            #     x = x.reshape(x.shape[0], -1)
            #     print(x.shape)
            #     print(x)
            #     exit(1)

        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0
        top5 = 0

        iters, extra = divmod(x.shape[0], batch_size)

        for i in range(iters):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            # print("Gonna predict")
            y = self.predict(tx)
            # Compute Top-5 metric
            for j in range(len(tx)):
                top5 += tt[j] in np.argsort(y[j].ravel())[-5:]
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        if extra > 0:
            tx = x[iters * batch_size:]
            tt = t[iters * batch_size:]
            # print("Gonna EXTRA predict")
            y = self.predict(tx)
            # Compute Top-5 metric
            for j in range(len(tx)):
                top5 += tt[j] in np.argsort(y[j].ravel())[-5:]
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        # return (acc / x.shape[0], top5 / x.shape[0])
        return (acc, top5)

    def numerical_gradient(self, x, t):
        """求梯度 (数值微分)
        Parameters
        ----------
        x : 输入数据
        t : 教师标签
        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        def loss_w(w): return self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w,
                                                       self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w,
                                                       self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度 (误差反向传播法)
        Parameters
        ----------
        x : 输入数据
        t : 教师标签
        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            # print(key, val.shape)
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl", _t=sp.posit8):
        # with open(file_name, 'rb') as f:
        #     params = pickle.load(f)
        # for key, val in params.items():
        #     print(type(val.item(0)), val.item(0))
        #     self.params[key] = val

        params, names = read_params(10, file_name)

        for i, p in enumerate(params):
            k = names[i]
            # print(names[i])
            if i == 0 or i == 2:
                # Convert Conv weights into FCKK(NCWH) format
                p = p.transpose(3, 2, 0, 1)
            p_shape = p.shape
            # print(p_shape)
            p = p.ravel()
            aux = np.empty_like(p, dtype=_t)

            for j in range(p.size):
                aux[j] = _t(float(p[j]))

            # print(p_shape)
            self.params[k] = np.reshape(aux, p_shape)
            # print(type(self.params[k].item(0)), self.params[k].item(0))
            # self.params[key] = p

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine3', 'Affine4', 'Affine5']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# Actual training/inference process
posit_t = sp.posit8
# # test_init = 0
# # test_end = 10
# total_batch = int(sys.argv[1])
# batch_n = sys.argv[2]
# N = int(10000/total_batch)
# test_init = N*int(batch_n)
# test_end = N*(int(batch_n)+1)
# print('Test Images: ', test_init, test_end)

# Read data
(_, _), (X_test, y_test) = mnist.load_data()
# (10000, 28 * 28)
# [test_init:test_end] # NCWH format
X_test = X_test.reshape(-1, 1, 28, 28)[:100]
y_test = y_test[:100]  # [test_init:test_end]

# Pad images (LeNet only accepts 32x32 input imgs.)
X_test = np.pad(X_test, [(0, 0), (0, 0), (2, 2), (2, 2)], 'constant')

X_test = ((X_test-127.5)/127.5)  # .astype(np_type)

# CONVERT INPUTS TO POSIT
aux = np.empty_like(X_test, dtype=posit_t)
print('Converting imgs to posit format...')
for i in range(X_test.size):
    aux.flat[i] = posit_t(X_test.flat[i])

X_test = aux
print(type(X_test.item(0)))

# exit(1)

# Generate model
network = SimpleConvNet(input_dim=(1, 32*32),
                        output_size=10, weight_init_std=0.01, _t=posit_t)

network.load_params(file_name='../data/lenet5/mnist/posit32.ckpt', _t=posit_t)
print('NN generated!')


def acc(batch):
    # print('Type is ', type(X_test.item(0)))
    k = 1  # 125 # Batch size
    i = k*batch
    return network.accuracy(X_test[i:i+k], y_test[i:i+k])


tic = time.time()

# a = network.accuracy(X_test, y_test)
print("Performing inference stage (multiprocessing)")
try:
    # mp.set_start_method('spawn')
    cores = min(cpu_count(), 70)
    pool = Pool(processes=cores)  # on system number of processors
    print(f'Using {cores} workers')

    # , chunksize=len(y_test)//cores
    a = pool.map(acc, iterable=range(len(y_test)))
finally:  # To make sure processes are closed in the end, even if errors happen
    pool.close()
    pool.join()

toc = time.time()
# print(return_dict)
# a = a.get()
a = np.array(a)
# a = np.array(return_dict)
a = np.sum(a, axis=0)/len(y_test)

print(a)

hist = {}
# Adding list as value
hist["val_acc"] = []
hist["top5"] = []
hist["val_acc"].append(a[0])
hist["top5"].append(a[1])

s = int(toc - tic)
m, s = divmod(s, 60)
h, m = divmod(m, 60)

print(f"Elapsed time: {h} h, {m} min, {s} sec.")

# # Save parameters
# network.save_params("data/params.pkl")
# print("Saved Network Parameters!")

# Save training results
files_path = './train_results/lenet5/'+'mnist/'
results_path = files_path + 'posit8_quire.csv'

zd = zip(*hist.values())

with open(results_path, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(hist.keys())
    writer.writerows(zd)

# Show results
s = int(toc-tic)
m, s = divmod(s, 60)
h, m = divmod(m, 60)
days, h = divmod(h, 24)

body = f'The Inference phase with data type {posit_t} on TensorFlow (LeNet MNIST + Quire) has finished after {h} h, {m} min, {s} sec!\n\nThe ACC and Top-5 are:\n{hist}'
print(body)
