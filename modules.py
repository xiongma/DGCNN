#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/3
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf

__all__ = ['atrous_conv1d', 'attention_encoder', 'get_embedding', 'create_bias_initializer', 'create_kernel_initializer']

def create_kernel_initializer(stddev=0.01):
    """
    create kernel truncate normal initializer
    :param stddev: standard deviation
    :return: tensorflow truncate normal initializer
    """
    return tf.truncated_normal_initializer(stddev=stddev)

def create_bias_initializer(type='conv'):
    """
    create bias constant initializer
    :param type: if type equals to conv, return 0 constant initializer, if equals to dense, return 1 constant initializer
    :return: tensorflow constant initializer
    """
    if type == 'dense': return tf.constant_initializer(1)
    else: return tf.constant_initializer(0)

def atrous_conv1d(X, window=3, dilation=1, scope='atrous_conv1d'):
    """
    expansion of convolution: X + tf.multiply((Conv1D1(X)-X), Conv1D2(X))
    :param X: embedding
    :param dilation: the size of expansion
    :param window: the size of kernel length
    :param scope: scope name
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        filters = X.shape.as_list()[-1]

        # conv1
        conv1 = tf.layers.conv1d(X,
                                 filters,
                                 window,
                                 dilation_rate=dilation,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=create_bias_initializer('conv'))
        conv1 = tf.subtract(conv1, X)

        # conv2
        conv2 = tf.layers.conv1d(X,
                                 filters,
                                 window,
                                 dilation_rate=dilation,
                                 activation=tf.sigmoid,
                                 padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=create_bias_initializer('conv'))

        conv = X + tf.multiply(conv1, conv2)

        # mask
        conv = tf.where(tf.equal(X, 0), X, conv)

        return conv

def attention_encoder(X, scope='attention_encoder'):
    """
    attention encoder, see more detail in https://www.cnblogs.com/callyblog/p/11111493.html
    :param X: inputs
    :param scope: scope name, default attention_encoder
    :return: sum of every time context vector
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        time_step = X.shape.as_list()[-2]

        attention = tf.layers.dense(X,
                                    64,
                                    use_bias=False,
                                    activation=tf.tanh,
                                    name='tanh_fully',
                                    kernel_initializer=create_kernel_initializer())
        attention = tf.layers.dense(attention,
                                    time_step,
                                    use_bias=False,
                                    name='softmax_fully',
                                    kernel_initializer=create_kernel_initializer())

        # mask
        padding_num = -2 ** 32 + 1 # multiply max number, let 0 index of timestep equal softmax 0
        masks = tf.sign(tf.reduce_sum(tf.abs(X), axis=-1))  # [N, T]
        masks = tf.tile(tf.expand_dims(masks, axis=1), [1, time_step, 1])  # [N, T, T]
        paddings = tf.ones_like(masks) * padding_num
        attention = tf.where(tf.equal(masks, 0), paddings, attention)

        # softmax
        attention = tf.nn.softmax(attention)

        # attention * X
        outputs = tf.matmul(attention, X) # [N, T, H]
        outputs = tf.reduce_sum(outputs, axis=1) # [N, H]

        return outputs

def get_embedding(vec, maxlen1, masks1, masks2):
    """
    get bert embedding
    :param vec: bert vec instance
    :param maxlen1: question max length
    :param masks1: question masks
    :param masks2: evidence masks
    :return: question embedding and evidence embedding by bert
    """
    embedding = vec.sequence_output
    splits = tf.split(embedding, num_or_size_splits=2, axis=0)
    ques_embedd, evidence_embedd = splits[0], splits[1]

    ques_embedd = ques_embedd[:, :maxlen1, :]
    masks1 = masks1[:, :maxlen1]

    ques_embedd = ques_embedd * tf.cast(tf.expand_dims(masks1, axis=-1), dtype=tf.float32)
    evidence_embedd = evidence_embedd * tf.cast(tf.expand_dims(masks2, axis=-1), dtype=tf.float32)

    return ques_embedd, evidence_embedd