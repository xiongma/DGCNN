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

__all__ = ['split_inputs', 'calc_num_batches', 'import_tf', 'save_variable_specs', 'concat_inputs']

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

def concat_inputs(xs, ys):
    """
    concat input ids, input masks, segment ids
    :param xs: xs, input ids, input mask, segment ids
    :param ys: ys, input ids, input mask, segment ids
    :return: input ids, contain question and evidence input ids
             input masks, contain question and evidence input masks
             segment ids, contain question and evidence segment ids
    """
    input_ids = tf.concat([xs[0], ys[0]], axis=0)
    input_masks = tf.concat([xs[1], ys[1]], axis=0)
    segment_ids = tf.concat([xs[2], ys[2]], axis=0)

    return input_ids, input_masks, segment_ids
