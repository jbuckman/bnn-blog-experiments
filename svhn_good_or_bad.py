import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import urllib.request
import pathlib

# Training Parameters
learning_rate = 0.0001
num_steps = 20000
batch_size = 256
eval_batch_size = 4096
display_step = 1000
α = .01

# Network Parameters
input_shape = [32, 32, 3] # SVHN data input (img shape: 32*32*3)
dropout = 1. # Dropout, probability to keep units

def init_conv_weights_xavier(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def init_fc_weights_xavier(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def init_biases(shape, name):
    return tf.Variable(tf.constant(0.0, shape=shape), name=name)

def conv_layer(input_tensor,  # The input or previous layer
               params,  # Params
               pooling):  # Average pooling

    weights, biases = params

    # Create the TensorFlow operation for convolution, with S=1 and zero padding
    activations = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], 'SAME') + biases

    # layernorm
    layernormer = tf.keras.layers.LayerNormalization()
    activations = layernormer(activations)

    # Rectified Linear Unit (ReLU)
    #         activations = tf.nn.relu(activations)
    activations = tf.nn.leaky_relu(activations, alpha=0.10)
    #         activations = tf.contrib.layers.maxout(activations, num_units=num_filters)

    # pooling layer
    if pooling:
        # Create a pooling layer with F=2, S=1 and zero padding
        #             activations = tf.nn.max_pool(activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        activations = tf.nn.avg_pool(activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    return activations

def flatten_layer(input_tensor):
    """ Helper function for transforming a 4D tensor to 2D
    """
    # Get the shape of the input_tensor.
    input_tensor_shape = input_tensor.get_shape()

    # Calculate the volume of the input tensor
    num_activations = input_tensor_shape[1:4].num_elements()

    # Reshape the input_tensor to 2D: (?, num_activations)
    input_tensor_flat = tf.reshape(input_tensor, [-1, num_activations])

    # Return the flattened input_tensor and the number of activations
    return input_tensor_flat, num_activations


def fc_layer(input_tensor,  # The previous layer,
             params,  # Params
             layernorm=False,
             relu=False):

    weights, biases = params

    # Calculate the layer activation
    activations = tf.matmul(input_tensor, weights) + biases

    if layernorm:
        # layernorm
        layernormer = tf.keras.layers.LayerNormalization()
        activations = layernormer(activations)

    if relu:
        activations = tf.nn.leaky_relu(activations, alpha=0.10)

    return activations

def get_params(name):
    # Store layers weight & bias
    weights = {
        'c1': init_conv_weights_xavier([5, 5, 3, 32], name+'-wc1'),
        'c2': init_conv_weights_xavier([5, 5, 32, 32], name+'-wc2'),
        'c3': init_conv_weights_xavier([5, 5, 32, 64], name+'-wc3'),
        'c4': init_conv_weights_xavier([5, 5, 64, 64], name+'-wc4'),
        'c5': init_conv_weights_xavier([5, 5, 64, 128], name+'-wc5'),
        'c6': init_conv_weights_xavier([5, 5, 128, 128], name+'-wc6'),
        'c7': init_conv_weights_xavier([5, 5, 128, 128], name+'-wc7'),
        'f1': init_fc_weights_xavier([2048, 256], name+'-wf1'),
        'f2': init_fc_weights_xavier([256, 256], name+'-wf2'),
        'out': init_fc_weights_xavier([256, 10], name+'-wout'),
    }

    biases = {
        'c1': init_biases([1, 32], name+'-bc1'),
        'c2': init_biases([1, 32], name+'-bc2'),
        'c3': init_biases([1, 64], name+'-bc3'),
        'c4': init_biases([1, 64], name+'-bc4'),
        'c5': init_biases([1, 128], name+'-bc5'),
        'c6': init_biases([1, 128], name+'-bc6'),
        'c7': init_biases([1, 128], name+'-bc7'),
        'f1': init_biases([1, 256], name+'-bf1'),
        'f2': init_biases([1, 256], name+'-bf2'),
        'out': init_biases([1, 10], name+'-bout'),
    }
    return weights, biases

def conv_net(input, params, p_keep):
    weights, biases = params

    # Conv Block 1
    conv_1 = conv_layer(input, (weights["c1"], biases["c1"]), pooling=False)
    conv_2 = conv_layer(conv_1, (weights["c2"], biases["c2"]), pooling=True)
    drop_block1 = tf.nn.dropout(conv_2, p_keep)  # Dropout

    # Conv Block 2
    conv_3 = conv_layer(drop_block1, (weights["c3"], biases["c3"]), pooling=False)
    conv_4 = conv_layer(conv_3, (weights["c4"], biases["c4"]), pooling=True)
    drop_block2 = tf.nn.dropout(conv_4, p_keep)  # Dropout

    # Conv Block 3
    conv_5 = conv_layer(drop_block2, (weights["c5"], biases["c5"]), pooling=False)
    conv_6 = conv_layer(conv_5, (weights["c6"], biases["c6"]), pooling=False)
    conv_7 = conv_layer(conv_6, (weights["c7"], biases["c7"]), pooling=True)
    flat_tensor, num_activations = flatten_layer(tf.nn.dropout(conv_7, p_keep))  # Dropout

    # Fully-connected 1
    fc_1 = fc_layer(flat_tensor, (weights["f1"], biases["f1"]), relu=True, layernorm=True)
    drop_fc2 = tf.nn.dropout(fc_1, p_keep)  # Dropout

    # Fully-connected 2
    fc_2 = fc_layer(drop_fc2, (weights["f2"], biases["f2"]), relu=True, layernorm=True)

    # Parallel softmax layers
    logits = fc_layer(fc_2, (weights["out"], biases["out"]))

    return logits

def compare_logits_to_answer(logits, answer):
    return tf.cast(tf.equal(tf.argmax(logits, -1), answer), tf.float32)

def flatten_params(params):
    return list(params[0].values()) + list(params[1].values())

def tf_flatten_params(params):
    param_list = flatten_params(params)
    return tf.concat([tf.reshape(p, [-1]) for p in param_list], 0)

def l2_distance(a, b): return tf.sqrt(tf.reduce_sum((a - b) ** 2.))

if __name__ == '__main__':
    # Mode
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', choices=['good', 'bad'])
    args = parser.parse_args()
    mode = args.mode

    # Dataset loading
    pathlib.Path("/tmp/data/svhn/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./models").mkdir(parents=True, exist_ok=True)
    if "train_32x32.mat" not in os.listdir("/tmp/data/svhn/"):
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                                   "/tmp/data/svhn/train_32x32.mat")
    if "test_32x32.mat" not in os.listdir("/tmp/data/svhn/"):
        urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                                   "/tmp/data/svhn/test_32x32.mat")
    svhn_train = sio.loadmat('/tmp/data/svhn/train_32x32.mat')
    svhn_test = sio.loadmat('/tmp/data/svhn/test_32x32.mat')

    clean_X = svhn_train["X"].transpose(3, 0, 1, 2)
    clean_Y = svhn_train["y"] % 10
    corrupted_X = svhn_test["X"].transpose(3, 0, 1, 2)
    corrupted_Y = np.random.randint(0,10,[corrupted_X.shape[0], 1])

    # tf Graph input
    X = tf.placeholder(tf.float32, [None] + input_shape)
    Y = tf.placeholder(tf.int64, [None, 1])
    M = tf.placeholder(tf.float32, [None, 1]) # mask for eval
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    if mode == 'good':
        X_data = clean_X
        Y_data = clean_Y
        M_data = np.ones_like(clean_Y)
    else:
        X_data = np.concatenate([clean_X, corrupted_X], 0)
        Y_data = np.concatenate([clean_Y, corrupted_Y], 0)
        M_data = np.concatenate([np.ones_like(clean_Y), np.zeros_like(corrupted_Y)], 0)

    # make params
    params = get_params("good")
    flat_params = tf_flatten_params(params)

    # compute prob under gaussian prior
    standard_gaussian_prior = tf.compat.v1.distributions.Normal(loc=tf.zeros_like(flat_params), scale=tf.ones_like(flat_params))
    params_log_pdfs = standard_gaussian_prior.log_prob(flat_params)

    # set up predictions for points and curves
    prediction = conv_net(X, params, keep_prob)

    # compute losses
    point_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=tf.one_hot(Y, 10)[:, 0, :]))
    reg_loss = tf.reduce_mean(-params_log_pdfs)

    # mask out losses appropriately, and add evaluations
    loss_op = point_loss + α*reg_loss
    if mode == "good":
        evaluate = {"prior_log_prob": tf.reduce_sum(params_log_pdfs),
                    "prior_log_prob_per_param": tf.reduce_mean(params_log_pdfs),
                    "accuracy": tf.reduce_mean(compare_logits_to_answer(prediction, Y[:, 0]))}
    else:
        evaluate = {"prior_log_prob": tf.reduce_sum(params_log_pdfs),
                    "prior_log_prob_per_param": tf.reduce_mean(params_log_pdfs),
                    "clean_accuracy": tf.reduce_sum(M[:, 0] * compare_logits_to_answer(prediction, Y[:, 0])) / tf.reduce_sum(M[:, 0]),
                    "corrupted_accuracy": tf.reduce_sum((1. - M[:, 0]) * compare_logits_to_answer(prediction, Y[:, 0])) / tf.reduce_sum(1. - M[:, 0]), }

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    sess = tf.Session()

    # Run the initializer
    sess.run(init)

    # Load models
    losses = []
    for step in range(1, num_steps+1):
        idx = np.random.randint(0, X_data.shape[0], [batch_size])
        batch_x, batch_y, batch_m = X_data[idx], Y_data[idx], M_data[idx]
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y, M: batch_m, keep_prob: dropout})
        losses.append(loss)
        if step % display_step == 0 or step == 1:
            idx = np.random.randint(0, X_data.shape[0], [eval_batch_size])
            batch_x, batch_y, batch_m = X_data[idx], Y_data[idx], M_data[idx]
            ev = sess.run(evaluate, feed_dict={X: batch_x, Y: batch_y, M: batch_m, keep_prob: dropout})
            print(f"STEP {step} LOSS {np.mean(losses)} EVALUATIONS {ev}")
            losses = []
    print("\nOptimization Finished!")
