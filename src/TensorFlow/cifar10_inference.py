# -*- coding: utf-8 -*-
"""
![CifarNet Architecture]
Source: Alex Krizhevsky, 2013
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from tensorflow.contrib.layers import flatten
import sys
import csv
import os
import time

# Remove this for further evaluation
np.random.seed(1)
tf.set_random_seed(2)

# Load CIFAR10 Dataset
data_set = 'CIFAR-10'
print("Dataset is: ", data_set)
(_, _), (X_test, y_test) = cifar10.load_data()
# somehow y_train comes as a 2D nx1 matrix
# y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

# assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

# X_train = X_train[:500]
# y_train = y_train[:500]
# X_test = X_test[:10]
# y_test = y_test[:10]

print("\nImage Shape: {}\n".format(X_test[0].shape))
# print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

"""## Setup TensorFlow
The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
"""

EPOCHS = 30
BATCH_SIZE = 128
print('Total epochs:', EPOCHS)

# Set Posit data types
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if sys.argv[1] == 'posit32':
        eps = 1e-8
        posit = np.posit32
        tf_type = tf.posit32
    elif sys.argv[1] == 'posit16':
        eps = 1e-4
        posit = np.posit16
        tf_type = tf.posit16
    elif sys.argv[1] == 'posit8':
        eps = 0.015625
        posit = np.posit8
        tf_type = tf.posit8
else:
    eps = 1e-8
    data_t = 'float32'
    posit = np.float32
    tf_type = tf.float32

# confirm dtype
print("\nType is: ", data_t)

# Normalize data
# X_train = (X_train/255.).astype(posit) # [0,1] normalization
# X_test = (X_test/255.).astype(posit)
# X_train = ((X_train-127.5)/127.5).astype(posit)  # [-1,1] normalization
X_test = ((X_test-127.5)/127.5).astype(posit)

# Standarize data
# mu_X = np.mean(X_train, axis=(0, 1, 2))
# std_X = np.std(X_train, axis=(0, 1, 2))
# X_train = ((X_train-mu_X)/std_X).astype(posit)
# X_test = ((X_test-mu_X)/std_X).astype(posit)

print("Input data type: {}".format(type(X_test[0, 0, 0, 0])))

# https://github.com/tensorflow/tensorflow/blob/24f578cd66bfc3ec35017fc77e136e43c4b74742/tensorflow/python/kernel_tests/lrn_op_test.py
# https://www.kaggle.com/sarvesh278/cnn-and-batch-normalization-in-tensorflow
# https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf_type),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=tf_type),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=posit(0.99))

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            # Modified tensorflow/python/ops/gen_nn_ops.py to include posit dtypes
                            # Modified tensorflow/python/training/moving_averages.py to include posit dtypes
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


"""## Implementation of CifarNet
Implements the [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)

# Input
The CifarNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since CIFAR10 images are RGB, C is 3 in this case.

# Output
Return the forwarded prediction - logits.
"""


def Convnet(x, training):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x64.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 3, 64), mean=mu, stddev=sigma, dtype=tf_type))
    conv1_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[
                         1, 1, 1, 1], padding='SAME') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 32x32x64. Output = 16x16x64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # Local Response Normalization
    # Use BatchNorm instead

    # Batch Normalization
    # conv1 = batch_norm(conv1, 64, training)
    # conv1 = tf.compat.v1.layers.BatchNormalization()(conv1)

    # Layer 2: Convolutional. Output = 16x16x64.
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 64, 64), mean=mu, stddev=sigma, dtype=tf_type))
    conv2_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[
                         1, 1, 1, 1], padding='SAME') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Local Response Normalization
    #conv2 = tf.cast(conv2, tf.float32)
    #norm2 = tf.nn.lrn(conv2, 4, bias=posit(1.0), alpha=posit(0.001 / 9.0), beta=posit(0.75))
    #norm2 = tf.cast(norm2, tf_type)

    # Batch Normalization
    # conv2 = batch_norm(conv2, 64, training)

    # Pooling. Input = 16x16x64. Output = 8x8x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # Dropout
    #conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 8x8x64. Output = 4096.
    fc0 = flatten(conv2)
    dim = fc0.get_shape()[1].value

    # Layer 3: Fully Connected. Input = 4096. Output = 384.
    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(dim, 384), mean=mu, stddev=sigma, dtype=tf_type))
    fc1_b = tf.Variable(tf.zeros(384, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    #fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 384. Output = 192.
    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(384, 192), mean=mu, stddev=sigma, dtype=tf_type))
    fc2_b = tf.Variable(tf.zeros(192, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    #fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Linear layer(WX + b). Input = 192. Output = 10.
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(192, 10), mean=mu, stddev=sigma, dtype=tf_type))
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

    # out = tf.identity(logits+0)
    # return out


"""## Features and Labels

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.
"""

x = tf.placeholder(tf_type, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
# keep_prob = tf.placeholder(tf_type, (None))
training = tf.placeholder(tf.bool)

"""## Training Pipeline
Create a training pipeline that uses the model to classify CIFAR10 data.
"""

rate = posit(0.001)

logits = Convnet(x, training)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate, beta1=posit(
    0.9), beta2=posit(0.999), epsilon=posit(eps))
# training_operation = optimizer.minimize(loss_operation)

"""## Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.
"""

correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
in_top5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), y, k=5)
top5_operation = tf.reduce_mean(tf.cast(in_top5, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_top5 = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, top5 = sess.run([accuracy_operation, top5_operation], feed_dict={
                                  x: batch_x, y: batch_y, training: False})
        total_accuracy += (accuracy * len(batch_x))
        total_top5 += (top5 * len(batch_x))
    return (total_accuracy / num_examples, total_top5 / num_examples)


def get_top5(X_data, y_data):
    num_examples = len(X_data)
    total_top5 = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        top5 = sess.run(top5_operation, feed_dict={
                        x: batch_x, y: batch_y, training: False})
        total_top5 += (top5 * len(batch_x))
    return (total_top5 / num_examples)


def validate(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        _loss, _acc = sess.run([loss_operation, accuracy_operation], feed_dict={
                               x: batch_x, y: batch_y, training: False})
        total_accuracy += (_acc * len(batch_x))
        total_loss += (_loss * len(batch_x))
    return (total_loss / num_examples, total_accuracy / num_examples)


hist = {}
# Adding list as value
hist["val_acc"] = []
hist["top5"] = []

"""## Remove all training nodes
Only inference step will be computed with pretrained weights.
Therefore, training nodes are not neccessary at all.

If want to continue training the network (e.g. on transfer learning) comment this line.
"""
tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())

data_dir = '../data/CIFAR10/'
model_name = data_dir + 'posit32.ckpt'
# model_name = data_dir + 'float32.ckpt'

assert os.path.exists(
    data_dir), "The directory %s does not exist!" % data_dir


"""## Load and cast the Pre-trained Model before Evaluate
https://stackoverflow.com/a/47077472

Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
"""
tic = time.time()
with tf.Session() as sess:
    previous_variables = [
        var_name for var_name, _
        in tf.contrib.framework.list_variables(model_name)]
    # print(previous_variables)
    sess.run(tf.global_variables_initializer())
    restore_map = {}
    for variable in tf.global_variables():
        if variable.op.name in previous_variables:
            # print(variable.op.name)
            var = tf.contrib.framework.load_variable(
                model_name, variable.op.name)
            if(var.dtype == np.posit32):
                tf.add_to_collection('assignOps', variable.assign(
                    tf.cast(var, tf_type)))
            else:
                print('Var. found of type ', var.dtype)  # Log
                tf.add_to_collection('assignOps', variable.assign(var))
    sess.run(tf.get_collection('assignOps'))
    print('Pre-Trained Parameters loaded and casted as type', tf_type)

    print('Computing Acc. (Top-1) & Top-5...')
    val_acc, test_top5 = evaluate(X_test, y_test)
    hist["val_acc"].append(val_acc)
    hist["top5"].append(test_top5)
    print('Accuracy:', val_acc)
    print('Top-5:', test_top5)

toc = time.time()
# Save training results
results_dir = './train_results/CIFAR10/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)  # Unreachable(?)

zd = zip(*hist.values())

with open(results_dir + data_t + '.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(hist.keys())
    writer.writerows(zd)

with open(results_dir + 'top5.txt', "a+") as f:
    f.write("%s: %s\n" % (data_t, test_top5))

# Show results
s = int(toc-tic)
m, s = divmod(s, 60)
h, m = divmod(m, 60)
days, h = divmod(h, 24)

body = 'The inference phase (pre-trained weights) with data type %s on TensorFlow (%s) has finished after %s h, min, sec!\n\nThe results are:\n%s' % (
print(body)
