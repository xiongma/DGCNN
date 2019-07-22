#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/5
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import pandas as pd
import tensorflow as tf

from bert_vec import get_tokenizer
from utils import calc_num_batches

__all__ = ['get_batch']

def _load_data(fpath, maxlen1, maxlen2, tokenizer):
    """
    load data
    :param fpath: data path
    :param maxlen1: question max length
    :param maxlen2: evidence max length
    :param tokenizer: Bert tokenizer
    :return: questions, list, every element is input ids, input masks, segment ids
             evidences, list, every element is input ids, input masks, segment ids
             labels, list, every element is list which contain label, start index, end index
    """
    df = pd.read_excel(fpath).dropna(how='any').values.tolist()
    questions = []
    evidences = []
    labels = []
    for data in df:
        question = data[1]
        evidence = data[0]
        if not isinstance(question, str) or not isinstance(evidence, str):
            continue
        if len(question) > maxlen1 or len(question) < 5: continue
        if len(evidence) > maxlen2 or len(evidence) < 20: continue

        label = data[3]
        start = data[4]
        end = data[5]
        questions.append(question)
        evidences.append(evidence)
        labels.append([label, start, end])

    questions_ = []
    evidences_ = []
    from bert_vec import convert_single_example
    for question, evidence in zip(questions, evidences):
        input_ids1, input_mask1, segment_ids1 = convert_single_example(tokenizer, question, maxlen2)
        input_ids2, input_mask2, segment_ids2 = convert_single_example(tokenizer, evidence, maxlen2)
        questions_.append([input_ids1, input_mask1, segment_ids1])
        evidences_.append([input_ids2, input_mask2, segment_ids2])

    return questions_, evidences_, labels

def _generator_fn(questions, evidences, labels):
    """
    Generates training / evaluation data
    :param questions: questions, list
    :param evidences: evidences, list
    :param labels: labels, list, every element is [label, start_index, end_index]
    :return: generator: question ids
                        evidences ids
                        labels, label contain start index onehot, end index onehot
    """
    for question, evidence, label in zip(questions, evidences, labels):
        yield tuple(question), tuple(evidence), label

def _input_fn(questions, evidences, labels, batch_size, gpu_nums, maxlen2, shuffle=False):
    """
    distribute data to every batch
    :param questions: questions, list
    :param evidences: evidences, list
    :param labels: labels, list, every element is [label, start_index, end_index]
    :param batch_size: batch size, int
    :param gpu_nums: gpu numbers, int
    :param maxlen2: evidence max length, int
    :param shuffle: whether shuffle data, When train model, it's True
    :return: tensorflow dataset
    """
    shapes = (([maxlen2], [maxlen2], [maxlen2]), ([maxlen2], [maxlen2], [maxlen2]), ([3]))
    types = ((tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32), (tf.int32))
    dataset = tf.data.Dataset.from_generator(generator=_generator_fn,
                                             output_shapes=shapes,
                                             output_types=types,
                                             args=(questions, evidences, labels))

    # when training, the dataset will be shuffle
    # if shuffle:
    #     dataset = dataset.shuffle(buffer_size=batch_size*gpu_nums)

    dataset = dataset.repeat() # iterator forever
    dataset = dataset.batch(batch_size=batch_size*gpu_nums, drop_remainder=False)
    return dataset

def get_batch(fpath, maxlen1, maxlen2, batch_size, gpu_nums, bert_pre, shuffle=False):
    """
  Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    """
    tokenizer = get_tokenizer(bert_pre)
    questions, evidences, labels = _load_data(fpath, maxlen1, maxlen2, tokenizer)
    batches = _input_fn(questions, evidences, labels, batch_size, gpu_nums, maxlen2, shuffle=shuffle)
    num_batches = calc_num_batches(len(questions), batch_size * gpu_nums)
    return batches, num_batches, len(questions)