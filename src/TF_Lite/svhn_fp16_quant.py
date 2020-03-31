import sys
import numpy as np
import tensorflow as tf
import scipy.io as sio
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
data_set = 'SVHN'

saved_model_dir = './data/' + data_set + '/'
results_dir = './inference_results/' + data_set + '/'
test_location = 'dataset/' + saved_model_dir + 'test_32x32.mat'

# confirm Dataset
print("Dataset is: ", data_set)


def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:, :, :, i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    # for i in range(len(Y_test)):
    #     if Y_test[i]%10 == 0:
    #         Y_test[i] = 0
    # Y_test = to_categorical(Y_test,10)
    Y_test %= 10
    return (X_test, Y_test)


X_test, y_test = load_test_data()
# somehow y_train comes as a 2D nx1 matrix
y_test = y_test.reshape(y_test.shape[0])

assert(len(X_test) == len(y_test))

# Normalize data
X_test = ((X_test.astype('float32')-127.5) / 127.5)  # (60000, 32, 32, 3)

assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))


# Convert to quantized tf.lite model

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
graph_def_file = saved_model_dir+"frozen_model.pb"
input_arrays = ["inputs"]
output_arrays = ["logits"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays)

tf.logging.set_verbosity(tf.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
#open(saved_model_dir+'quantized_model.tflite', 'wb').write(tflite_model)
tflite_models_dir = pathlib.Path(saved_model_dir)
tflite_model_file = tflite_models_dir/'quant_model_FP16.tflite'
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

eval_data = X_test

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
f.write("FP16 Quantization: %s\n" % (t5 / len(eval_data)))
f.close()

f = open(results_dir + 'FP16_quant.txt', "a+")
f.write("Top-1: %s\nTop-5: %s\n" % (acc / len(eval_data), t5 / len(eval_data)))
f.close()
