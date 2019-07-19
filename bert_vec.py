#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
date: 2019/7/16
mail: cally.maxiong@gmail.com
blog: http://www.cnblogs.com/callyblog/
'''
import os

import tensorflow as tf

from bert import modeling, tokenization

__all__ = ['convert_single_example', 'BertVec', 'get_tokenizer']

def get_tokenizer(model_dir):

    return tokenization.FullTokenizer(os.path.join(model_dir, 'vocab.txt'))

def convert_single_example(tokenizer, text_a, max_len):
    """
    convert text a and text b to id, padding [CLS] [SEP]
    :param tokenizer: bert tokenizer
    :param text_a: text a
    :param max_len: max sequence length
    :return: input ids, input mask, segment ids
    """
    tokens = []
    segment_ids = []
    text_a = tokenizer.tokenize(text_a)[: max_len-2] # [CLS] [SEP]

    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in text_a:
        tokens.append(token)
        segment_ids.append(0)
    segment_ids.append(0)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids

class BertVec:
    def __init__(self, pre_bert, input_ids, input_mask, segment_ids):
        self.ckpt = tf.train.get_checkpoint_state(pre_bert).all_model_checkpoint_paths[-1]
        self.bert_config = modeling.BertConfig.from_json_file(os.path.join(pre_bert, 'bert_config.json'))

        # init graph
        self._init_graph(input_ids, input_mask, segment_ids)

    def _init_graph(self, input_ids, input_mask, segment_ids):
        """
        init bert graph
        :param input_ids: tensorflow placeholder input ids
        :param input_mask: tensorflow placeholder input mask
        :param segment_ids: tensorflow placeholder segment ids
        """
        self._model = modeling.BertModel(config=self.bert_config,
                                         is_training=False,
                                         input_ids=input_ids,
                                         input_mask=input_mask,
                                         token_type_ids=segment_ids,
                                         use_one_hot_embeddings=False)

        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, self.ckpt)
        tf.train.init_from_checkpoint(self.ckpt, assignment_map)

    @property
    def pooled_output(self):
        """
        get Bert pooled output
        """
        return self._model.pooled_output

    @property
    def sequence_output(self):
        """
        get Bert sequence output
        :return:
        """
        return self._model.sequence_output
