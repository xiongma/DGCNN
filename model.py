#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/3
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import tensorflow as tf

from modules import atrous_conv1d, attention_encoder, get_token_embeddings, positional_encoding, \
    create_kernel_initializer, create_bias_initializer

from utils import noam_scheme, split_inputs

__all__ = ['DGCNN']

class DGCNN:
    def __init__(self, hp, zero_pad=True):
        self.embedding = get_token_embeddings(hp.vocab_size, hp.num_units, zero_pad=zero_pad)
        self.num_units = hp.num_units
        self.hp = hp

    def question(self, xs, mask):
        """
        DGCNN question encoder, you can see more detail in https://www.cnblogs.com/callyblog/p/11111493.html
        :param xs:
        :return:
        """
        with tf.variable_scope('question', reuse=tf.AUTO_REUSE):
            ques_embedd = tf.nn.embedding_lookup(self.embedding, xs)

            ques_conv = atrous_conv1d(ques_embedd, mask, window=3, dilation=1)

            attention = attention_encoder(ques_conv, mask)

            return attention

    def evidence(self, ys, mask, attention, training):
        """
        evidence encoding, decoding, see https://www.cnblogs.com/callyblog/p/11111493.html
        :param ys: evidences
        :param attention: question encoding embedding
        :param training: whether train, if True, use dropout
        :return: p_global probability [N]
                 p_start probability [N, maxlen]
                 p_end probability [N, maxlen]
        """
        with tf.variable_scope('evidence', reuse=tf.AUTO_REUSE):
            # lookup
            evidence_embedd = tf.nn.embedding_lookup(self.embedding, ys)
            # position embedding
            positional_embedding = positional_encoding(evidence_embedd, self.hp.maxlen2)

            attention = tf.tile(tf.expand_dims(attention, 1), [1, self.hp.maxlen2, 1]) # [N, T, H1]

            # concat position embedding, question attention embedding
            evidence_embedd = tf.concat([evidence_embedd, attention, positional_embedding], axis=-1) # [N, T, H+maxlen+H]

            # feature fusion
            evidence_conv = atrous_conv1d(evidence_embedd, mask, dilation=1, window=1, scope='feature_fusion')

            # conv
            evidence_conv = atrous_conv1d(evidence_conv, mask, dilation=1, window=3, scope='atrous_conv1_dilation1')
            evidence_conv = atrous_conv1d(evidence_conv, mask, dilation=2, window=3, scope='atrous_conv1_dilation2')
            evidence_conv = atrous_conv1d(evidence_conv, mask, dilation=4, window=3, scope='atrous_conv1_dilation4')

            evidence_conv = tf.concat([evidence_conv, attention], axis=2)
            ques_mater_attention = attention_encoder(evidence_conv, mask) # [N, H]

            # dropout
            ques_mater_attention = tf.layers.dropout(ques_mater_attention, rate=self.hp.dropout_rate, training=training)
            evidence_conv = tf.layers.dropout(evidence_conv, rate=self.hp.dropout_rate, training=training)

            # p global
            p_global = tf.layers.dense(ques_mater_attention, 1, activation=tf.sigmoid, name='p_global',
                                       kernel_initializer=create_kernel_initializer(),
                                       bias_initializer=create_bias_initializer('dense')) # [N, 1]

            # p start
            p_start = tf.layers.dense(evidence_conv, 1, activation=tf.sigmoid, name='p_start_sigmoid',
                                      kernel_initializer=create_kernel_initializer(),
                                      bias_initializer=create_bias_initializer('dense')) # [N, T, 1]

            # p end
            p_end = tf.layers.dense(evidence_conv, 1, activation=tf.sigmoid, name='p_end_sigmoid',
                                    kernel_initializer=create_kernel_initializer(),
                                    bias_initializer=create_bias_initializer('dense')) # [N, T, 1]

            p_global_ = tf.expand_dims(p_global, axis=-1)
            p_start = p_global_ * p_start # [N, T, 1]
            p_end = p_global_ * p_end # [N, T, 1]

            return p_global, p_start, p_end

    def train(self, xs, ys, labels, total_steps):
        """
        train DGCNN model
        :param xs: question
        :param ys: evidence
        :param labels: labels, contain global, start, end
        :param total_steps: total train steps
        :return: train op, loss, global step, tensorflow summary
        """
        tower_grads = []
        global_step = tf.train.get_or_create_global_step()
        global_step_ = global_step * self.hp.gpu_nums

        warmup = False
        if self.hp.warmup_rate > 0.0:
            warmup = True
        warmup_steps = int(total_steps * self.hp.warmup_rate)

        lr = noam_scheme(global_step_, warmup_steps, total_steps, self.hp.lr, warmup)
        optimizer = tf.train.AdamOptimizer(lr)

        losses = []
        datas = split_inputs(self.hp.gpu_nums, xs, ys, labels)

        with tf.variable_scope(tf.get_variable_scope()):
            for no in range(self.hp.gpu_nums):
                with tf.device("/gpu:%d" % no):
                    with tf.name_scope("tower_%d" % no):
                        ques_atten = self.question(datas[0][0][no], datas[0][1][no])
                        _, p_start, p_end = self.evidence(datas[1][0][no], datas[1][1][no], ques_atten, True)

                        tf.get_variable_scope().reuse_variables()
                        loss = self._calc_loss(datas[2][no], p_start, p_end)
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

    def eval(self, xs, ys, labels):
        """
        evaluate model, just use one gpu to evaluate
        :param xs: questions
        :param ys: evidences
        :param labels: labels
        :return: answer indexes, loss, tensorflow summary
        """
        x, x_mask = xs
        y, y_mask = ys
        ques_atten = self.question(x, x_mask)
        p_global, p_start, p_end = self.evidence(y, y_mask, ques_atten, False)

        # loss
        loss = self._calc_loss(labels, p_start, p_end)

        tf.summary.scalar('eval_loss', loss)
        summaries = tf.summary.merge_all()

        return p_global, p_start, p_end, loss, summaries

    def _calc_loss(self, labels, p_start, p_end):
        """
        calculate loss
        :param labels: labels, contain p_global, p_start, p_end
        :param p_start: predicted p_start
        :param p_end: predicted p_end
        :return: p_global loss + p_start loss + p_end loss
        """
        # start loss
        p_start_true = labels[:, 1]
        p_start_true = tf.expand_dims(tf.one_hot(p_start_true, depth=self.hp.maxlen2), axis=-1)
        # p_start_true = tf.one_hot(tf.one_hot(p_start_true, depth=self.hp.maxlen2, dtype=tf.int32), depth=2)
        p_start_loss = self._focal_loss(p_start, p_start_true)

        # end loss
        p_end_true = labels[:, 2]
        p_end_true = tf.expand_dims(tf.one_hot(p_end_true, depth=self.hp.maxlen2), axis=-1)
        # p_end_true = tf.one_hot(tf.one_hot(p_end_true, depth=self.hp.maxlen2, dtype=tf.int32), depth=2)
        p_end_loss = self._focal_loss(p_end, p_end_true)

        loss = p_start_loss + p_end_loss

        return loss

    def _focal_loss(self, pred, y, alpha=0.25, gamma=2):
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
        per_entry_cross_ent = tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent, axis=[1, 2]))
        # per_entry_cross_ent = tf.reduce_sum(per_entry_cross_ent)
        return per_entry_cross_ent

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
