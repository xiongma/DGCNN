#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/3
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import numpy as np
import tensorflow as tf

__all__ = ['get_token_embeddings', 'atrous_conv1d', 'attention_encoder', 'noam_scheme', 'positional_encoding',
           'create_bias_initializer', 'create_kernel_initializer']

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

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
        return embeddings

def atrous_conv1d(X, mask, window=3, dilation=1, scope='atrous_conv1d'):
    """
    expansion of convolution
    :param X: embedding
    :param dilation: the size of expansion
    :param window: the size of kernel length
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
        mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=2)
        conv = conv * mask

        return conv

def attention_encoder(X, mask, scope='attention_encoder'):
    """
    attention encoder, see more detail in https://www.cnblogs.com/callyblog/p/11111493.html
    :param X: inputs
    :param mask: mask
    :param scope: scope name, default attention_encoder
    :return: sum of every time context vector
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        X_ = X
        X = tf.layers.dense(X, X.get_shape().as_list()[-1], use_bias=False, activation=tf.tanh)
        X = tf.layers.dense(X, 1, use_bias=False)
        mask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)
        pad_num = tf.multiply(1.0 - mask, tf.cast(1e30, dtype=tf.float32))
        X = X - pad_num
        X = tf.nn.softmax(X, 1)
        outputs = tf.reduce_sum(X * X_, 1)

        return outputs

def positional_encoding(inputs, maxlen, masking=True, scope='positional_encoding'):
    '''Sinusoidal Positional_Encoding. See https://www.cnblogs.com/callyblog/p/11111493.html
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''
    E = inputs.shape.as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i % 2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return outputs

def noam_scheme(d_model, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    d_model: encoder and decoder embedding
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return d_model ** -0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)