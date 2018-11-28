# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import collections
import numpy as np
import random
# 控制batch的生产
data_index = 0

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
    unk_count  = len(words) - reduce(
        lambda x, y : x + y, map(lambda x:x[1], freq_words))# 更新UNK的数量
    count = [('UNK', unk_count)]
    count.extend(freq_words)
    # dictionary key 是词， value是index
    dictionary = dict(zip(map(lambda x :x[0], count), range(len(count)))) 
    data = map(lambda x: dictionary.get(x, 0), words)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    """
    Inputs:
    - batch_size: 每个batch所含的样本数量
    - num_skips: 在一个window下（2 * skip_window + 1）产生context词的数量
    - skip_window window半边大小
    Returns a tuple of:
    - data_index: array，每个词的数值表示 shape(batch_size, )
    - labels: array，样本的标注，即某个context的数值表示 shape(batch_size, )
    """
    global data # word 的数值表示
    global data_index
    assert batch_size % num_skips == 0 
    assert num_skips <= 2 * skip_window 
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips): #i 是每一次的窗口滑动
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1 #每次挪动一个词
    # 将data index 倒回一点点，使得整个数据采样比较均匀。
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

if __name__ == '__main__':
    with open('small_text') as f:
        data = tf.compat.as_str(f.read()).split()
    # 词数
    vocabulary_size = 100

    data, count, dictionary, reverse_dictionary = build_dataset(data, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(
            batch[i], 
            reverse_dictionary[batch[i]], 
            '->', 
            labels[i, 0], 
            reverse_dictionary[labels[i, 0]])
    
    # batch 的参数
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.

    # 验证相关的参数
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # 开始建图
    graph = tf.Graph()
    with graph.as_default():
        
        with tf.name_scope('inputs'):
            train_input = tf.placeholder(tf.int32, shape=[batch_size])
            train_label = tf.placeholder(tf.int32, shape=[batch_size])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            emb_vec = tf.nn.embedding_lookup(embeddings, train_input)
        
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=0.001))

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([vocabulary_size]))