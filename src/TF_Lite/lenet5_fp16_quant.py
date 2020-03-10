import sys
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

tf.enable_eager_execution()
tf.lite.constants.FLOAT16


np.random.seed(1)
tf.set_random_seed(2)

# Load Dataset
if len(sys.argv) > 1:
    data_set = sys.argv[1]
    if sys.argv[1] == 'mnist':
        d_set = mnist
    elif sys.argv[1] == 'fashion_mnist':
        d_set = fashion_mnist
else:
    data_set = 'mnist'
    d_set = mnist

saved_model_dir = './train_results/lenet5/' + data_set + '/' 

# confirm Dataset
print("Dataset is: ", data_set)

(X_train, y_train), (X_test, y_test) = d_set.load_data()
X_train = np.expand_dims(X_train, axis=3)  # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=3)  # (10000, 28, 28, 1)

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))


# Normalize data
# X_train = (X_train.astype('float32') / 255.0).reshape(-1,28, 28, 1)  # (60000, 28, 28, 1)
# X_test = (X_test.astype('float32') / 255.0).reshape(-1,28, 28, 1)  # (10000, 28, 28, 1)
X_train = ((X_train.astype('float32')-127.5)/127.5)
X_test = ((X_test.astype('float32')-127.5)/127.5)

# convert to quantized tf.lite model

images = tf.cast(X_train, tf.float32)
mnist_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)

## construct and provide a representative dataset
## this is used to get the dynamic range of activations
def representative_data_gen():
    for input_value in mnist_ds.take(100):
        yield [input_value]


# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
graph_def_file = saved_model_dir+"frozen_model.pb"
input_arrays = ["inputs"]
output_arrays = ["logits"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)

tf.logging.set_verbosity(tf.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
#open(saved_model_dir+'quantized_model.tflite', 'wb').write(tflite_model)
tflite_models_dir = pathlib.Path(saved_model_dir)
tflite_model_file = tflite_models_dir/'quant_model_FP16.tflite'
tflite_model_file.write_bytes(tflite_model)

# Load the quantized tf.lite model and test

#interpreter = tf.lite.Interpreter(model_path=saved_model_dir+'quantized_model.tflite')
interpreter = tf.lite.Interpreter(
        model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

acc = 0
t5 = 0

# eval_data = np.array(X_test * 255 - 128, dtype=np.int8)
# eval_data = np.array(X_test * 255, dtype = np.uint8)
eval_data = X_test

for i in range(eval_data.shape[0]):
    # image = eval_data[i].reshape(1,28,28,1)
    image = eval_data[i].reshape(1, 32, 32, 1)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    five_pred = tf.argsort(prediction[0], direction='DESCENDING',)[:5]
    if ((tf.reduce_sum(tf.cast(tf.equal(y_test[i], five_pred), tf.int8)))):
        t5 += 1
        if (y_test[i]) == np.argmax(prediction):
            acc += 1

print('Post-training integer quantization accuracy: ' + str(acc / len(eval_data)))
print('Post-training integer quantization Top-5: ' + str(t5 / len(eval_data)))

f = open(saved_model_dir + 'top5.txt', "a+")
f.write("FP16 Quantization: %s\n" % (t5 / len(eval_data)))
f.close()

f = open(saved_model_dir + 'FP16_quant.txt', "a+")
f.write("Top-1: %s\nTop-5: %s\n" % (acc / len(eval_data), t5 / len(eval_data)))
f.close()
