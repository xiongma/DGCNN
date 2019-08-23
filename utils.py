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
from tqdm import tqdm
import numpy as np
import jieba

__all__ = ['label_smoothing', 'split_inputs', 'calc_num_batches', 'import_tf', 'save_variable_specs', 'noam_scheme',
           'get_hypotheses']

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
    split_data = []
    for arg in args[:2]:
        sub = []
        for a in arg:
            sub.append(tf.split(a, num_or_size_splits=gpu_nums, axis=0))
        split_data.append(sub)
    split_data.append(tf.split(args[-1], num_or_size_splits=gpu_nums, axis=0))
    return split_data

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

def get_hypotheses1(num_batches, sess, tensor, handle_placehoder, handle, idx2token, logdir):
    f = open(os.path.join(logdir, 'hypotheses'), 'w', encoding='utf-8')
    accuracies, f1s = [], []
    for _ in tqdm(range(num_batches)):
        p_gloabls, p_starts, p_ends, evidences, labels = sess.run(tensor, feed_dict={handle_placehoder: handle})

        for p_gloabl, p_start, p_end, evidence, label in zip(p_gloabls, p_starts, p_ends, evidences, labels):
            l = label[0]
            start = label[1]
            end = label[2]

            if l == 0:
                p_start_max = np.argmax(p_start, axis=0)
                p_end_max = np.argmax(p_end, axis=0)
                if sum(p_start_max) == 0 and sum(p_end_max) == 0 and p_gloabl < 0.5:
                    accuracies.append(1)
                    f1s.append(1)
                else:
                    accuracies.append(0)
                    f1s.append(0)
            if l == 1:
                p_start_max = np.argmax(p_start[:, 1], axis=-1)
                p_end_max = np.argmax(p_end[:, 1], axis=-1)

                answer_true = evidence[start: end + 1]
                answer_true = ''.join([idx2token[c] for c in answer_true])

                if p_start_max > p_end_max:
                    accuracies.append(0)
                    f1s.append(0.0)
                    f.write('pred: NONE\n true: {}\n'.join(answer_true))
                else:
                    answer_pred = evidence[p_start_max: p_end_max+1]
                    answer_pred = ''.join([idx2token[c] for c in answer_pred])
                    f.write('pred: {}\n true: {}\n'.format(answer_pred, answer_true))
                    if answer_pred == answer_true:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)

                    words_pred = jieba.lcut(answer_pred)
                    words_true = jieba.lcut(answer_true)

                    num = 0
                    for word in words_pred:
                        if word in words_true:
                            num += 1

                    f1s.append(num / len(words_true))

    assert len(accuracies) == len(f1s), 'Some Error occur'

    return sum(accuracies) / len(accuracies), sum(f1s) / len(f1s)

def get_hypotheses(num_batches, sess, tensor, handle_placehoder, handle, idx2token, logdir):
    f = open(os.path.join(logdir, 'hypotheses'), 'w', encoding='utf-8')
    accuracies, f1s = [], []
    for _ in tqdm(range(num_batches)):
        p_gloabls, p_starts, p_ends, evidences, labels = sess.run(tensor, feed_dict={handle_placehoder: handle})

        for p_gloabl, p_start, p_end, evidence, label in zip(p_gloabls, p_starts, p_ends, evidences, labels):
            l = label[0]
            start = label[1]
            end = label[2]

            # if l == 0:
            #     p_start_max = np.argmax(p_start, axis=0)
            #     p_end_max = np.argmax(p_end, axis=0)
            #     if sum(p_start_max) == 0 and sum(p_end_max) == 0 and p_gloabl < 0.5:
            #         accuracies.append(1)
            #         f1s.append(1)
            #     else:
            #         accuracies.append(0)
            #         f1s.append(0)
            if l == 1:
                p_start_max = np.argmax(p_start[:, 0], axis=-1)
                p_end_max = np.argmax(p_end[:, 0], axis=-1)

                answer_true = evidence[start: end + 1]
                answer_true = ''.join([idx2token[c] for c in answer_true])

                if p_start_max > p_end_max:
                    accuracies.append(0)
                    f1s.append(0.0)
                    f.write('pred: NONE\n true: {}\n'.format(answer_true))
                else:
                    answer_pred = evidence[p_start_max: p_end_max+1]
                    answer_pred = ''.join([idx2token[c] for c in answer_pred])
                    f.write('pred: {}\n true: {}\n'.format(answer_pred, answer_true))
                    if answer_pred == answer_true:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)

                    words_pred = jieba.lcut(answer_pred)
                    words_true = jieba.lcut(answer_true)

                    num = 0
                    for word in words_pred:
                        if word in words_true:
                            num += 1

                    f1s.append(num / len(words_true))

    assert len(accuracies) == len(f1s), 'Some Error occur'

    return sum(accuracies) / len(accuracies), sum(f1s) / len(f1s)


