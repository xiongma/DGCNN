#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/4
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

# import numpy as np
# import tensorflow as tf
#
# a = [1, 0, 0, 0, 0]
# a = tf.convert_to_tensor(a, dtype=tf.int32)
# b = [1, 0, 0, 1, 1]
# b = tf.convert_to_tensor(b, dtype=tf.int32)
#
# c = tf.reduce_sum(tf.to_int32(tf.equal(a, b))) / tf.to_int32(tf.equal(a, b)).shape.as_list()[-1]
#
# d = [[1, 2], [4, 3], [5, 6]]
# d = tf.convert_to_tensor(d, dtype=tf.int32)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
#
# print(sess.run(d[:, 1]))
# v = sess.run(tf.argmax(d[:, 1], axis=0))
# print(v)

a = """刘嘉玲的老公是梁朝伟
"""
print(a[7:10])