# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import collections


def build_dataset(words, n_words):
    """
    Inputs:
    - words: 输入的词列表，包含重复
    - n_words: 最n_words个常用的词，其他为UNK
        Returns a tuple of:
    - data: words的数值表示，其中UNK为0， 类型为list
    - count: [('UNK', -1), ('the', 12), ('of', 9), ('a', 6), ...] 长度为 n_words
    - dictionary: 类型dict，词 -> 数值
    - reversed_dictionary: 类型dict, 数值 -> 词
    """
    freq_words = collections.Counter(words).most_common(n_words - 1)
    unk_count  = len(words) - reduce(lambda x, y : x + y, map(lambda x:x[1], freq_words))# 更新UNK的数量
    count = [('UNK', unk_count)]
    count.extend(freq_words)
    dictionary = dict(zip(map(lambda x :x[0], count), range(len(count)))) # key 是词， value是index
    data = map(lambda x: dictionary.get(x, 0), words)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

if __name__ == '__main__':
    with open('small_text') as f:
        data = tf.compat.as_str(f.read()).split()
    data, count, dictionary, reverse_dictionary = build_dataset(data, 100)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])