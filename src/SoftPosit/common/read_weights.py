import numpy as np
# import softposit as sp
import tensorflow as tf

import os


def read_params(N_vars, file_name):

    # files_path = './data/lenet/'
    # directory = os.path.dirname(files_path)
    # model_name = files_path + 'posit32.ckpt'

    # assert os.path.exists(
    #     directory), "The directory %s does not exist!" % directory

    # N_vars = 8
    weights_list = [None] * N_vars
    var_names = [None] * N_vars

    with tf.Session() as sess:
        previous_variables = [
            var_name for var_name, _ in
            tf.contrib.framework.list_variables(file_name)]

        # Keep only weights - no optimization vars are needed
        aux_list = previous_variables.copy()
        for i, e in reversed(list(enumerate(aux_list))):
            if ('/' in e) or ('beta' in e):
                previous_variables.pop(i)

        # print(previous_variables)
        for i, var_name in enumerate(previous_variables[:N_vars]):
            var = tf.contrib.framework.load_variable(
                file_name, var_name)
            # print(var_name, var.dtype)
            # print(var.shape)
            # if(var.dtype == np.posit32):
            #     if var_name == 'Variable':
            #         fc1_W = var
            #     elif var_name == 'Variable_1':
            #         fc1_b = var
            #     elif var_name == 'Variable_2':
            #         fc2_W = var
            #     elif var_name == 'Variable_3':
            #         fc2_b = var
            #     elif var_name == 'Variable_4':
            #         fc3_W = var
            #     elif var_name == 'Variable_5':
            #         fc3_b = var
            #     elif var_name == 'Variable_6':
            #         fc4_W = var
            #     elif var_name == 'Variable_7':
            #         fc4_b = var

            #     else:
            #         print("Unexpected value found")
            #         exit(1)

            # var is a numpy ndarray
            weights_list[i] = var
            # var_names[i] = var_name

    # var_names = previous_variables[:N_vars]

    # print(var_names)
    # print(weights_list)
    for i in range(N_vars//2):
        var_names[2*i] = 'W' + str(i + 1)
        var_names[2*i+1] = 'b' + str(i + 1)

    # return ([fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b, fc4_W, fc4_b], var_names)
    return (weights_list, var_names)
