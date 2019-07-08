#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/4
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import logging
import os

import tensorflow as tf

__all__ = ['label_smoothing', 'split_inputs', 'calc_num_batches', 'import_tf', 'save_variable_specs']

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = tf.cast(tf.shape(inputs)[-1], tf.float32)  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)

def split_inputs(*args):
    """
    split inputs
    :param gpu_nums: gpu numbers
    :return: split inputs by gpu numbers
    """
    gpu_nums = args[0]
    args = args[1:]
    return [tf.split(arg, num_or_size_splits=gpu_nums, axis=0) for arg in args]

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def import_tf(gpu_list):
    """
    import tensorflow, set tensorflow graph load device
    :param gpu_list: GPU list
    :return: tensorflow instance
    """
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)

    return tf

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    import tensorflow as tf
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w', encoding='utf-8') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")