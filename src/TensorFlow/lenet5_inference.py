# -*- coding: utf-8 -*-
"""
![LeNet Architecture](lenet.png)
Source: Yan LeCun
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.contrib.layers import flatten
import sys
import csv
import os
import time

# Remove this for further evaluation
np.random.seed(1)
tf.set_random_seed(2)

# Load Dataset
if len(sys.argv) > 2:
    data_set = sys.argv[2]
    if sys.argv[2] == 'mnist':
        d_set = mnist
    elif sys.argv[2] == 'fashion_mnist':
        d_set = fashion_mnist
else:
    data_set = 'mnist'
    d_set = mnist

# confirm Dataset
print("Dataset is: ", data_set)

(_, _), (X_test, y_test) = d_set.load_data()
# X_train = np.expand_dims(X_train, axis=3)  # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=3)  # (10000, 28, 28, 1)

# assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

# X_train = X_train[:500]
# y_train = y_train[:500]
# X_test = X_test[:500]
# y_test = y_test[:500]

print()
print("Image Shape: {}".format(X_test[0].shape))
print()
# print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

"""The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.

However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.

In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).

You do not need to modify this section.
"""

# Pad images with 0s
# X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_test[0].shape))

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
    elif sys.argv[1] == 'float32':
        eps = 1e-8
        posit = np.float32
        tf_type = tf.float32
else:
    eps = 1e-8
    data_t = 'posit32'
    posit = np.posit32
    tf_type = tf.posit32

print()
# confirm dtype
print("Type is: ", posit)

# Normalize data
# X_train = (X_train/255.).astype(posit)
# X_test = (X_test/255.).astype(posit)
# X_train = ((X_train-127.5)/127.5).astype(posit)
X_test = ((X_test-127.5)/127.5).astype(posit)

print("Input data type: {}".format(type(X_test[0, 0, 0, 0])))

"""## Implementation of LeNet-5 - Build a copy if the network specifying the data type
Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

This is the only cell you need to edit.
# Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

# Architecture
**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

**Layer 3: Fully Connected.** This should have 120 outputs.

**Activation.** Your choice of activation function.

**Layer 4: Fully Connected.** This should have 84 outputs.

**Activation.** Your choice of activation function.

**Layer 5: Fully Connected (Logits).** This should have 10 outputs.

# Output
Return the result of the 2nd fully connected layer.
"""


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf_type))
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

    # Dropout
    #conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf_type))
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
    # out = tf.identity(conv2)

    # Dropout
    #conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(400, 120), mean=mu, stddev=sigma, dtype=tf_type))
    fc1_b = tf.Variable(tf.zeros(120, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    # out = tf.identity(fc0+0)

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    #fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(120, 84), mean=mu, stddev=sigma, dtype=tf_type))
    fc2_b = tf.Variable(tf.zeros(84, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    #fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(84, 10), mean=mu, stddev=sigma, dtype=tf_type))
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
    # return out


"""## Features and Labels
Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.
"""

x = tf.placeholder(tf_type, (None, 32, 32, 1), name='inputs')
y = tf.placeholder(tf.int32, (None), name='labels')
# keep_prob = tf.placeholder(tf_type, (None))

"""## Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.
"""

rate = posit(0.001)

logits = tf.identity(LeNet(x), name="logits")
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
                                  x: batch_x, y: batch_y})
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
        top5 = sess.run(top5_operation, feed_dict={x: batch_x, y: batch_y})
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
                               x: batch_x, y: batch_y})
        total_accuracy += (_acc * len(batch_x))
        total_loss += (_loss * len(batch_x))
    return (total_loss / num_examples, total_accuracy / num_examples)


"""## Load the Pre-Trained Model
Run the training data through the training pipeline to train the model.

# Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.
"""

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

data_dir = '../data/lenet5/'+data_set+'/'
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
    print('Pre-trained parameters loaded and casted as type', tf_type)

    print('Computing Acc. (Top-1) & Top-5...')
    val_acc, test_top5 = evaluate(X_test, y_test)
    hist["val_acc"].append(val_acc)
    hist["top5"].append(test_top5)
    print('Accuracy:', val_acc)
    print('Top-5:', test_top5)

toc = time.time()
# Save training results
results_dir = './train_results/lenet5/'+data_set+'/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)  # Unreachable

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

body = 'The inference phase (pre-trained weights) with data type %s on TensorFlow (%s) has finished after %s h, min, sec!\n\nThe results are:\n%s' % (posit, data_set, (days, h, m, s), test_top5, hist)
print(body)
