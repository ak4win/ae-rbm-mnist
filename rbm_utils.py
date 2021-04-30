""" Utility functions for creating custom RBMs"""

import tensorflow as tf
import numpy as np
#====================================================================================================
#
#====================================================================================================
def apply_weights(x, weights):
    # First approach: simple matrix multiplication
    output = tf.linalg.matmul(x, weights, transpose_a=False, transpose_b=False)

    # Second approach: convolutions
    
    return output
#====================================================================================================
#
#====================================================================================================
def forward_pass(x, weights):
    """
    Multiply the input by each of the weights and concatenate the results together

    arguments:
    x -- a tf.Tensor object that holds a matrix of x vals
    weights -- a tf.Tensor object that holds the weight vals. Can also be Tensorarray if multiple weights.

    returns;
    output -- a concatenated tensor the outputs of the individual sub-calculations
    """

    # As later code can only handle tf.TensorArray, convert weights to one if is only single
    if tf.is_tensor(weights):
        _weights = weights
        weights = tf.TensorArray(_weights.dtype, size=1)
        weights = weights.write(0, _weights)

    # 
    output = None
    for i in range(weights.size()):
        weight = weights.read(i)
        # weight = tf.transpose(weight)
        tmp_output = apply_weights(x, weight)
        if output is None:
            output = tmp_output
        else:
            output = tf.concat((output, tmp_output), axis=1)
    return output

#====================================================================================================
#
#====================================================================================================
def split_tensor(tensor, n_subparts):
    """
    Split a tensor into a number of subparts

    parameters:
    tensor -- the object that needs to be divided into subparts
    n_subparts -- the number of parts. must satsify: tensor.shape[0] % n_subparts == 0.

    """
    tf_array = tf.TensorArray(
        dtype=tf.float32,
        size=n_subparts
    )
    return tf_array.write(0, tensor)
    # tensor = tf.squeeze(tensor)
    # assert tf.rank(tensor) == 1, 'input tensor has to be of rank 1'
    # assert tensor.shape[0] % n_subparts == 0, 'tensor has to be dividable by num of subparts'

    # parts = tf.split(tensor, num_or_size_splits=n_subparts.numpy(), axis=0)
    # tf_array = tf.TensorArray(
    #     dtype=tf.float32,
    #     size=n_subparts
    # )
    # for i, part in enumerate(parts): tf_array = tf_array.write(i, tf.expand_dims(part, axis=0))
    # return tf_array
#====================================================================================================
#
#====================================================================================================
def backward_pass(x, weights):
    """
    Propagate the signal backwards.

    parameters:
    x -- the former output that needs to be propagated backwards now
    weights -- a tf.tensorArray of all the weights that are to be used

    returns:
    output -- 
    """

    # As later code can only handle tf.TensorArray, convert weights to one if is only single
    if tf.is_tensor(weights):
        _weights = weights
        weights = tf.TensorArray(_weights.dtype, size=1)
        weights = weights.write(0, _weights)

    splits = split_tensor(x, weights.size())

    # 
    outputs = tf.TensorArray(
        dtype=tf.float32,
        size=weights.size(),
        dynamic_size=True,
    )
    for i in range(weights.size()):
        weight = weights.read(i)
        weight = tf.transpose(weight)
        curr_split = splits.read(i)

        tmp_output = apply_weights(curr_split, weight)
        outputs = outputs.write(i, tmp_output)
    return outputs
#====================================================================================================
# 
#====================================================================================================
def loss(reconstruction, label):
    if (reconstruction == label).numpy().all(): print('GOT ZERO ')
    if np.isclose(reconstruction.numpy(), label.numpy()).all(): print('GOT IS CLOSE')
    return np.mean(((reconstruction - label)**2))
# def loss(prediction, label):
#     assert prediction.dtype == tf.float32, f'wrong dtype for prediction {prediction.dtype}'
#     assert label.dtype == tf.float32, f'wrong dtype for label {prediction.dtype}'

#     diff = tf.subtract(prediction, label)
#     sq = tf.square(diff)
#     return tf.cast(tf.reduce_mean(sq), tf.float32)