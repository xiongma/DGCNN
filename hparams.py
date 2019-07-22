#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/5
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    ## dataset files
    parser.add_argument('--train', default='data/WebQA/test_ann.xlsx', help="data for train")
    parser.add_argument('--eval', default='data/WebQA/test_ir.xlsx', help="data for evaluation")
    parser.add_argument('--bert_pre', default='bert_pre/', help="vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=8, type=int, help='the size of train batch')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='the size of eval batch')

    parser.add_argument('--lr', default=0.0005, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/2", help="log directory")
    parser.add_argument('--num_epochs', default=5, type=int)
    # model
    parser.add_argument('--maxlen1', default=32, type=int, help="maximum length of a question")
    parser.add_argument('--maxlen2', default=256, type=int, help="maximum length of a evidence")
    parser.add_argument('--dropout_rate', default=0.2, type=float, help="dropout rate")
    parser.add_argument('--gpu_nums', default=1, type=int,
                        help="gpu amount, which can allow how many gpus to train this model")