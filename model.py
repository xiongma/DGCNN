#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/3
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf

from modules import atrous_conv1d, attention_encoder, get_token_embeddings, positional_encoding
from utils import split_inputs, label_smoothing

__all__ = ['DGCNN']

class DGCNN:
    def __init__(self, hp, zero_pad=True):
        self.embedding = get_token_embeddings(hp.vocab_size, hp.num_units, zero_pad=zero_pad)
        self.num_units = hp.num_units
        self.hp = hp

    def question(self, xs):
        """
        DGCNN question encoder, you can see more detail in https://www.cnblogs.com/callyblog/p/11111493.html
        :param xs:
        :return:
        """
        with tf.variable_scope('question', reuse=tf.AUTO_REUSE):
            ques_embedd = tf.nn.embedding_lookup(self.embedding, xs)

            ques_conv = atrous_conv1d(ques_embedd, window=3, dilation=1, padding='SAME')

            attention = attention_encoder(ques_conv, dropout_rate=self.hp.dropout_rate)

            return attention

    def evidence(self, ys, attention, dropout_rate, maxlen, training):
        """
        evidence encoding, decoding, see https://www.cnblogs.com/callyblog/p/11111493.html
        :param ys: evidences
        :param attention: question encoding embedding
        :param dropout_rate: dropout rate
        :param maxlen: evidence max length
        :param training: whether train, if True, use dropout
        :return: p_global probability [N]
                 p_start probability [N, maxlen]
                 p_end probability [N, maxlen]
        """
        with tf.variable_scope('evidence', reuse=tf.AUTO_REUSE):
            evidence_embedd = tf.nn.embedding_lookup(self.embedding, ys)
            positional_embedding = positional_encoding(evidence_embedd, maxlen)

            attention = tf.tile(tf.expand_dims(attention, 1), [1, maxlen, 1]) # [N, T, H1]

            # concat position embedding, question attention embedding
            evidence_embedd = tf.concat([evidence_embedd, attention, positional_embedding], axis=-1) # [N, T, H+maxlen+H]

            # feature fusion
            evidence_conv = atrous_conv1d(evidence_embedd, dilation=1, window=1, scope='feature_fusion')

            # conv
            evidence_conv = atrous_conv1d(evidence_conv, dilation=1, window=3, scope='atrous_conv1_dilation1')
            evidence_conv = atrous_conv1d(evidence_conv, dilation=2, window=3, scope='atrous_conv1_dilation2')
            evidence_conv = atrous_conv1d(evidence_conv, dilation=4, window=3, scope='atrous_conv1_dilation4')

            ques_mater_attention = attention_encoder(evidence_conv, dropout_rate=dropout_rate) # [N, H]

            # dropout
            ques_mater_attention = tf.layers.dropout(ques_mater_attention, rate=dropout_rate, training=training)
            evidence_conv = tf.layers.dropout(evidence_conv, rate=dropout_rate, training=training)

            # fully connection
            p_global = tf.layers.dense(ques_mater_attention, 1, activation=tf.sigmoid, name='p_global') # [N, 1]
            p_start = tf.layers.dense(evidence_conv, 128, activation=tf.tanh, name='p_start_tanh') # [N, T, 128]
            p_start = tf.layers.dense(p_start, 2, activation=tf.sigmoid, name='p_start_sigmoid') # [N, T, 2]
            p_end = tf.layers.dense(evidence_conv, 128, activation=tf.tanh, name='p_end_tanh') # [N, T, 128]
            p_end = tf.layers.dense(p_end, 2, activation=tf.sigmoid, name='p_end_sigmoid') # [N, T, 2]

            p_start = tf.expand_dims(p_global, axis=-1) * p_start # [N, T, 2]
            p_end = tf.expand_dims(p_global, axis=-1) * p_end # [N, T, 2]

            return p_global, p_start, p_end

    def train_multi(self, xs, ys, labels):
        """
        train DGCNN model with multi GPUs
        :param xs: question
        :param ys: evidence
        :param labels: labels, contain global, start, end
        :return: train op, loss, global step, tensorflow summary
        """
        tower_grads = []
        global_step = tf.train.get_or_create_global_step()
        global_step_ = global_step * self.hp.gpu_nums

        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        losses = []
        datas = split_inputs(self.hp.gpu_nums, xs, ys, labels)

        with tf.variable_scope(tf.get_variable_scope()):
            for no in range(self.hp.gpu_nums):
                with tf.device("/gpu:%d" % no):
                    with tf.name_scope("tower_%d" % no):
                        ques_atten = self.question(datas[0][no])
                        p_global, p_start, p_end = self.evidence(datas[1][no], ques_atten, self.hp.dropout_rate,
                                                                 self.hp.maxlen2, True)

                        tf.get_variable_scope().reuse_variables()
                        loss = self._calc_loss(datas[2][no], p_global, p_start, p_end)
                        losses.append(loss)
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

        with tf.device("/cpu:0"):
            grads = self._average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
            loss = sum(losses) / len(losses)
            tf.summary.scalar("train_loss", loss)
            summaries = tf.summary.merge_all()

        return train_op, loss, summaries, global_step_

    def train_single(self, xs, ys, labels):
        """
        train DGCNN model with single GPU or CPU
        :param xs: question
        :param ys: evidence
        :param labels: labels, contain global, start, end
        :return: train op, loss, global step, tensorflow summary
        """
        global_step = tf.train.get_or_create_global_step()
        global_step_ = global_step * self.hp.gpu_nums

        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        ques_atten = self.question(xs)
        p_global, p_start, p_end = self.evidence(ys, ques_atten, self.hp.dropout_rate, self.hp.maxlen2, True)
        loss = self._calc_loss(labels, p_global, p_start, p_end)
        train_op = optimizer.minimize(loss, global_step=global_step)
        tf.summary.scalar("train_loss", loss)
        summaries = tf.summary.merge_all()

        return train_op, loss, summaries, global_step_

    def eval(self, xs, ys, labels):
        """
        evaluate model, just use one gpu to evaluate
        :param xs: questions
        :param ys: evidences
        :param labels: labels
        :return: answer indexes, loss, tensorflow summary
        """
        ques_atten = self.question(xs)
        p_global, p_start, p_end = self.evidence(ys, ques_atten, 1.0, self.hp.maxlen2, False)

        # loss
        loss = self._calc_loss(labels, p_global, p_start, p_end)

        # get answer
        p_start = tf.argmax(p_start, axis=1) # [N]
        p_end = tf.argmax(p_end, axis=1) # [N]

        p = tf.stack([p_start, p_end], axis=-1)

        tf.summary.scalar('eval_loss', loss)
        summaries = tf.summary.merge_all()

        return p, loss, summaries

    def _calc_loss(self, labels, p_global, p_start, p_end):
        """
        calculate loss
        :param labels: labels, contain p_global, p_start, p_end
        :param p_global: predicted p_global
        :param p_start: predicted p_start
        :param p_end: predicted p_end
        :return: p_global loss + p_start loss + p_end loss
        """
        # global loss
        p_global_true = labels[:, 0] # [N]
        p_global_true = label_smoothing(tf.one_hot(p_global_true, depth=2)) # [N, 2]
        p_global = tf.squeeze(tf.stack([1-p_global, p_global], axis=2), axis=1)
        p_global_loss = self._focal_loss(p_global, p_global_true)

        # start loss
        p_start_true = labels[:, 1]
        p_start_true = tf.one_hot(p_start_true, depth=self.hp.maxlen2, dtype=tf.int32)
        p_start_true = label_smoothing(tf.one_hot(p_start_true, depth=2))
        p_start_loss = self.focal_loss(p_start, p_start_true)

        # end loss
        p_end_true = labels[:, 2]
        p_end_true = tf.one_hot(p_end_true, depth=self.hp.maxlen2, dtype=tf.int32)
        p_end_true = label_smoothing(tf.one_hot(p_end_true, depth=2))
        p_end_loss = self.focal_loss(p_end, p_end_true)

        loss = p_start_loss + p_end_loss

        return loss

    def _focal_loss(self, pred, y, alpha=0.25, gamma=2):
        """
        focal loss
        :param pred: predicted
        :param y: true
        :param alpha: alpha, default 0.25
        :param gamma: gamma, default 2
        :return: focal loss
        """
        zeros = tf.zeros_like(pred, dtype=pred.dtype)
        pos_corr = tf.where(y > zeros, y-pred, zeros)
        neg_corr = tf.where(y > zeros, zeros, pred)
        fl_loss = -alpha * (pos_corr ** gamma) * tf.log(pred) - (1-alpha) * (neg_corr ** gamma) * tf.log(1.0 - pred)
        fl_loss = tf.reduce_mean(tf.reduce_sum(fl_loss, axis=1))
        return fl_loss

    def focal_loss(self, pred, y, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         pred: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         y: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        zeros = tf.zeros_like(pred, dtype=pred.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

        return tf.reduce_sum(per_entry_cross_ent)

    def _average_gradients(self, tower_grads):
        """
        average gradients of all gpu gradients
        :param tower_grads: list, each element is a gradient of gpu
        :return: be averaged gradient
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads
