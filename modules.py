#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/3
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import numpy as np
import tensorflow as tf

__all__ = ['get_token_embeddings', 'atrous_conv1d', 'attention_encoder', 'noam_scheme', 'positional_encoding']

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

def atrous_conv1d(X, filters, window=3, dilation=1, padding='SAME', scope='atrous_conv1d'):
    """
    expansion of convolution
    :param X: embedding
    :param dilation: the size of expansion
    :param window: the size of kernel length
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(X, filters, window, dilation_rate=dilation, padding=padding)
        conv1 = tf.add(conv1, 0-X)

        conv2 = tf.sigmoid(tf.layers.conv1d(X, filters, window, dilation_rate=dilation, padding=padding))

        conv = X + tf.multiply(conv1, conv2)

        # mask
        conv = tf.where(tf.equal(X, 0), X, conv)

        return conv

def attention_encoder(X, scope='attention_encoder'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        time_step = X.shape.as_list()[-2]

        attention = tf.layers.dense(X, time_step, use_bias=False, activation=tf.tanh, name='fully')
        gama = tf.get_variable(shape=[time_step, time_step], dtype=tf.float32, name='gama')
        attention = tf.matmul(attention, gama)

        # mask
        padding_num = -2 ** 32 + 1
        masks = tf.sign(tf.reduce_sum(tf.abs(X), axis=-1))  # [N, T]
        masks = tf.expand_dims(masks, axis=1)  # [N, T, T]
        paddings = tf.ones_like(masks) * padding_num
        attention = tf.where(tf.equal(masks), paddings, attention)

        attention = tf.nn.softmax(attention)

        outputs = tf.matmul(attention, X) # [N, T, H]
        outputs = tf.reduce_sum(outputs, axis=1) # [N, H]

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
    E = inputs.get_shape().as_list[-1]
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