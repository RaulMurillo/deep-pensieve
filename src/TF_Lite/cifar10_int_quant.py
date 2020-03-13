import sys
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import os

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

tf.enable_eager_execution()


np.random.seed(1)
tf.set_random_seed(2)

# Load Dataset
data_set = 'CIFAR10'

saved_model_dir = './data/' + data_set + '/'
results_dir = './inference_results/' + data_set + '/' 

# confirm Dataset
print("Dataset is: ", data_set)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# somehow y_train comes as a 2D nx1 matrix
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

# Normalize data
X_train = ((X_train-127.5) / 127.5)  # (60000, 32, 32, 3)
#X_test = (X_test.astype('float32') / 255.0)  # (60000, 32, 32, 3)

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))


# convert to quantized tf.lite model

images = tf.cast(X_train, tf.float32)
cifar_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)

## construct and provide a representative dataset
## this is used to get the dynamic range of activations
def representative_data_gen():
    for input_value in cifar_ds.take(100):
        yield [input_value]


# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
graph_def_file = saved_model_dir+"frozen_model.pb"
input_arrays = ["inputs"]
output_arrays = ["logits"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)

tf.logging.set_verbosity(tf.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(
    representative_data_gen)
# Ensure that the converted model is fully quantized
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
#open(saved_model_dir+'quantized_model.tflite', 'wb').write(tflite_model)
tflite_models_dir = pathlib.Path(saved_model_dir)
tflite_model_file = tflite_models_dir/'quant_model_INT8.tflite'
tflite_model_file.write_bytes(tflite_model)
print('TF Lite model saved!')

# Load the quantized tf.lite model and test

interpreter = tf.lite.Interpreter(
    model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

acc = 0
t5 = 0

eval_data = np.array(X_test - 128, dtype=np.int8)
# eval_data = np.array(X_test * 255, dtype = np.uint8)

for i in range(eval_data.shape[0]):
    image = eval_data[i].reshape(1, 32, 32, 3)

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

f = open(results_dir + 'top5.txt', "a+")
f.write("INT8 Quantization: %s\n" % (t5 / len(eval_data)))
f.close()

f = open(results_dir + 'INT8_quant.txt', "a+")
f.write("Top-1: %s\nTop-5: %s\n" % (acc / len(eval_data), t5 / len(eval_data)))
f.close()
