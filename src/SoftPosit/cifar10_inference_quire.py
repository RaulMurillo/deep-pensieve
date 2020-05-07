import numpy as np
import softposit as sp
from tensorflow.keras.datasets import cifar10
import sys
import os
import pickle
import csv
import time
from collections import OrderedDict
from common.layers import *
from common.matmul_quire import *
from common.read_weights import *
from multiprocessing import Pool, cpu_count

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

    def __init__(self, empty=True, input_dim=(3, 32, 32),
                 conv_param={'filter_num': 64,
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
                (filter_num, 3, filter_size, filter_size), dtype=_t)
            self.params['b1'] = np.empty((filter_num), dtype=_t)
            self.params['W2'] = np.empty(
                (filter_num, filter_num, filter_size, filter_size), dtype=_t)
            self.params['b2'] = np.empty((filter_num), dtype=_t)
            self.params['W3'] = np.empty((4096, 384), dtype=_t)
            self.params['b3'] = np.empty((384), dtype=_t)
            self.params['W4'] = np.empty((384, 192), dtype=_t)
            self.params['b4'] = np.empty((192), dtype=_t)
            self.params['W5'] = np.empty((192, output_size), dtype=_t)
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
                                           filter_stride, 2)  # Pad = 'SAME'
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           filter_stride, 2)
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
        # i = 0
        for layer in self.layers.values():
            # i += 1
            # print("$$$", layer, "$$$")
            # a = layer.W
            # print(layer, x.shape)
            # print(x)
            # print('******')
            # exit(1)
            x = layer.forward(x)
            # if (i >= 7):
            #     print("Forward!", layer)
            #     # a = layer.W
            #     # print(a.shape)
            #     # print(a)
            #     # x = x.transpose(0, 2, 3, 1)
            #     # x = x.reshape(x.shape[0], -1)
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

    def accuracy(self, x, t, batch_size=128):
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
            # print(y, np.argmax(y, axis=1), tt)
            # Compute Top-5 metric
            for j in range(len(tx)):
                top5 += tt[j] in np.argsort(y[j].ravel())[-5:]
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        # return (acc / x.shape[0], top5 / x.shape[0])
        return (acc, top5)

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）
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
        """求梯度（误差反向传播法）
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
dataset = 'CIFAR10'

# Read data
(_, _), (X_test, y_test) = cifar10.load_data()
# [test_init:test_end] # NCWH format
# assert((X_test.reshape(-1, 3, 32, 32) == X_test.transpose(0,3,1,2)).all())
# print(X_test.reshape(-1, 3, 32, 32))
# print('\n*******************\n************************\n')
# print(X_test.transpose(0,3,1,2))
# print('OK!')
# exit(1)

if(len(sys.argv) > 2):
    first_img = int(sys.argv[1])
    last_img = int(sys.argv[1]) + int(sys.argv[2])

    X_test = X_test.transpose(0, 3, 1, 2)[first_img:last_img]
    y_test = y_test.reshape(y_test.shape[0])[first_img:last_img]
    print(f"Inference with test images from {first_img} to {last_img}")
else:
    X_test = X_test.transpose(0, 3, 1, 2)  # [:10]
    # y_test = y_test#[:10]  # [test_init:test_end]
    # somehow y_train comes as a 2D nx1 matrix
    #y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    print("Inference with all test images")


# CIFAR-10 no need img. padding
# # Pad images (LeNet only accepts 32x32 input imgs.)
# # X_test = np.pad(X_test, [(0, 0), (0, 0), (2, 2), (2, 2)], 'constant')

X_test = ((X_test-127.5)/127.5)  # .astype(np_type)

# CONVERT INPUTS TO POSIT
aux = np.empty_like(X_test, dtype=posit_t)
print('Converting imgs to posit format...')
for i in range(X_test.size):
    aux.flat[i] = posit_t(X_test.flat[i])

X_test = aux
print('Working with type', type(X_test.item(0)))

# Generate model
network = SimpleConvNet(output_size=10, weight_init_std=0.01, _t=posit_t)

f_name = './data/'+ dataset + '/posit32.ckpt'
network.load_params(file_name=f_name, _t=posit_t)
print('NN generated!')


def acc(batch):
    # print('Type is ', type(X_test.item(0)))
    k = 32  # 125  # Batch size
    i = k*batch
    # print(y_test[i:i+k])
    # exit(1)
    return network.accuracy(x=X_test[i:i+k], t=y_test[i:i+k])


tic = time.time()
# print(y_test[:10])
# exit(1)

# a = network.accuracy(X_test[:10], y_test[:10])
print("Performing inference stage (multiprocessing)")
try:
    cores = min(cpu_count(), 80)
    pool = Pool(processes=cores)  # on system number of processors
    print(f'Using {cores} workers')

    # , chunksize=len(y_test)//cores
    a = pool.map(acc, iterable=range(len(y_test)//32))  # >= cores
finally:  # To make sure processes are closed in the end, even if errors happen
    pool.close()
    pool.join()

toc = time.time()
# a = a.get()
a = np.array(a)
a = np.sum(a, axis=0)/len(y_test)

print(a)

hist = {}
# Adding list as value
hist["val_acc"] = []
hist["top5"] = []
hist["val_acc"].append(a[0])
hist["top5"].append(a[1])

# # Save parameters
# network.save_params("data/params.pkl")
# print("Saved Network Parameters!")

# Save training results
results_path = './train_results/' + dataset +'/'
results_file = results_path+'posit8_'+str(first_img)+'_'+str(last_img)+'.csv'
if not os.path.exists(results_path):
    os.makedirs(results_path)  # Unreachable

zd = zip(*hist.values())

with open(results_file, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(hist.keys())
    writer.writerows(zd)

# Send results
send_mail = True
if(send_mail):
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(
        os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

    from other.mail import send_mail

    s = int(toc-tic)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    days, h = divmod(h, 24)

    subject = 'CNN Inference Quire compleated'
    body = f'The Inference phase with data type {posit_t} on TensorFlow ({dataset} + Quire) has finished after {h} h, {m} min, {s} sec!\n\nThe ACC and Top-5 are:\n{hist}'

    path = os.path.abspath('../other/credentials.txt')
    send_mail(subject=subject, mail_body=body, credentials=path)
