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

__all__ = ['split_inputs', 'calc_num_batches', 'import_tf', 'save_variable_specs', 'concat_inputs', 'noam_scheme']

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

def noam_scheme(global_step, num_warmup_steps, num_train_steps, init_lr, warmup=True):
    """
    decay learning rate
    if warmup > global step, the learning rate will be global_step/num_warmup_steps * init_lr
    if warmup < global step, the learning rate will be polynomial decay
    :param global_step: global steps
    :param num_warmup_steps: number of warm up steps
    :param num_train_steps: number of train steps
    :param init_lr: initial learning rate
    :param warmup: if True, it will warm up learning rate
    :return: learning rate
    """
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate,
                                                global_step,
                                                num_train_steps,
                                                end_learning_rate=0.0,
                                                power=1.0,
                                                cycle=False)

    if warmup:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate