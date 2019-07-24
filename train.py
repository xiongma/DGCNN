#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/5
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import logging
import os

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hparams import Hparams
from model import DGCNN
from utils import import_tf, save_variable_specs, get_hypotheses

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

# import tensorflow
gpu_list = [str(i) for i in list(range(hp.gpu_nums))]
tf = import_tf(gpu_list)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train,
                                                                hp.maxlen1,
                                                                hp.maxlen2,
                                                                hp.vocab,
                                                                hp.batch_size,
                                                                hp.gpu_nums,
                                                                shuffle=True)

eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval,
                                                             hp.maxlen1,
                                                             hp.maxlen2,
                                                             hp.vocab,
                                                             hp.eval_batch_size,
                                                             hp.gpu_nums,
                                                             shuffle=False)

handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(
    handle, train_batches.output_types, train_batches.output_shapes)

# create a iter of the correct shape and type
xs, ys, labels = iter.get_next()
logging.info('# init data')
training_iter = train_batches.make_one_shot_iterator()
val_iter = eval_batches.make_initializable_iterator()

logging.info("# Load model")
m = DGCNN(hp)

total_steps = hp.num_epochs * num_train_batches

# get op
train_op, train_loss, train_summaries, global_step = m.train(xs, ys, labels, total_steps)
p_global, p_start, p_end, eval_loss, eval_summaries = m.eval(xs, ys, labels)

token2idx, idx2token = load_vocab(hp.vocab)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    # Iterator.string_handle() get a tensor that can be got value to feed handle placeholder
    train_handle = sess.run(training_iter.string_handle())
    val_handle = sess.run(val_iter.string_handle())

    # start train
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps + 1)):
        _, _loss, _gs, _train_sum = sess.run([train_op, train_loss, global_step, train_summaries],
                                             feed_dict={handle: train_handle})
        summary_writer.add_summary(_train_sum, _gs)
        print(_loss)
        if _gs % 5000 == 0 and _gs != 0:
            logging.info("steps {} is done".format(_gs))

            logging.info("# test evaluation")
            sess.run(val_iter.initializer)  # initial val dataset
            _eval_summaries = sess.run(eval_summaries, feed_dict={handle: val_handle})
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info('# write hypotheses')
            accury, f1 = get_hypotheses(num_eval_batches, sess, [p_global, p_start, p_end, ys, labels], handle, val_handle, idx2token, hp.logdir)

            model_output = "trans_pointer%02dL%.2f%.2f" % (_gs, accury, f1)

            saver.save(sess, os.path.join(hp.logdir, model_output), global_step=_gs)