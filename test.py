#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/4
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''
import tensorflow as tf

a = [3]
a = tf.convert_to_tensor(a, dtype=tf.int32)
b = [0.1, 0.9]
b = tf.convert_to_tensor(b, dtype=tf.float32)

y_ = [[0.0, 1.0], [0.9, 0.1]]
y = [[0.0, 1.0], [0.7, 0.3]]
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-5, 1.0)))
cross_entropy1 = y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))

with tf.Session() as sess:
    print(sess.run(tf.stack(y, axis=-1)))