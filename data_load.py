#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/5
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''

import pandas as pd
import tensorflow as tf
from utils import calc_num_batches

__all__ = ['get_batch', 'load_vocab']

def _load_data(fpath, maxlen1, maxlen2):
    """
    load data
    :param fpath: data path
    :param maxlen1: question max length
    :param maxlen2: evidence max length
    :return: questions, list, every element is question
             evidences, list, every element is evidence
             labels, list, every element is list which contain label, start index, end index
    """
    df = pd.read_excel(fpath).dropna(how='any').values.tolist()
    questions = []
    evidences = []
    labels = []
    for data in df:
        question = data[1]
        evidence = data[0]
        if len(question) > maxlen1: continue
        if len(evidence) > maxlen2: continue

        label = data[3]
        start = data[4]
        end = data[5]
        questions.append(question.encode('utf-8'))
        evidences.append(evidence.encode('utf-8'))
        labels.append([label, start, end])

    return questions, evidences, labels

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>
    Returns
    two dictionaries.
    '''
    vocab = []
    with open(vocab_fpath, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.replace('\n', ''))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}

    return token2idx, idx2token

def _generator_fn(questions, evidences, labels, vocab_fpath):
    """
    Generates training / evaluation data
    :param questions: questions, list
    :param evidences: evidences, list
    :param labels: labels, list, every element is [label, start_index, end_index]
    :param vocab_fpath: vocab path, string
    :return: generator: question ids
                        evidences ids
                        labels, label contain start index onehot, end index onehot
    """
    token2idx, _ = load_vocab(vocab_fpath)
    for question, evidence, label in zip(questions, evidences, labels):
        tokens1 = [token2idx.get(token, token2idx['<unk>']) for token in list(question.decode('utf-8'))]
        tokens2 = [token2idx.get(token, token2idx['<unk>']) for token in list(evidence.decode('utf-8'))]

        yield tokens1, tokens2, label

def _input_fn(questions, evidences, labels, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=False):
    """
    distribute data to every batch
    :param questions: questions, list
    :param evidences: evidences, list
    :param labels: labels, list, every element is [label, start_index, end_index]
    :param vocab_fpath: vocab path, string
    :param batch_size: batch size, int
    :param gpu_nums: gpu numbers, int
    :param maxlen1: question max length, int
    :param maxlen2: evidence max length, int
    :param shuffle: whether shuffle data, When train model, it's True
    :return: tensorflow dataset
    """
    shapes = ([None]), ([None]), ([None])
    padded_shapes = ([maxlen1]), ([maxlen2]), ([3])
    types = (tf.int32), (tf.int32), (tf.int32)
    paddings = (0), (0), (0)
    dataset = tf.data.Dataset.from_generator(generator=_generator_fn,
                                             output_shapes=shapes,
                                             output_types=types,
                                             args=(questions, evidences, labels, vocab_fpath))

    # when training, the dataset will be shuffle
    if shuffle:
        dataset = dataset.shuffle(128*batch_size*gpu_nums)

    dataset = dataset.repeat() # iterator forever
    dataset = dataset.padded_batch(batch_size=batch_size*gpu_nums,
                                   padded_shapes=padded_shapes,
                                   padding_values=paddings).prefetch(1) # padding and batch

    return dataset

def get_batch(fpath, maxlen1, maxlen2, vocab_fpath, batch_size, gpu_nums, shuffle=False):
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
    questions, evidences, labels = _load_data(fpath, maxlen1, maxlen2)
    # for question in questions:
    #     if len(question) > maxlen1:
    #         print(1111)
    # for evidence in evidences:
    #     if len(evidence) > maxlen2: print(2222)
    batches = _input_fn(questions, evidences, labels, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=shuffle)
    num_batches = calc_num_batches(len(questions), batch_size * gpu_nums)
    return batches, num_batches, len(questions)