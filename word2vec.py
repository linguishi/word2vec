# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import tensorflow as tf
import collections
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE

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

# 打印低维词向量
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)

if __name__ == '__main__':
    with open('text8') as f:
        data = tf.compat.as_str(f.read()).split()
    # 词数
    vocabulary_size = 50000

    data, count, dictionary, reverse_dictionary = build_dataset(
        data, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(
            batch[i], 
            reverse_dictionary[batch[i]], 
            '->', 
            labels[i], 
            reverse_dictionary[labels[i]])
    
    # batch 的参数
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.
    learning_rate = 1.0
    train_steps = 100001

    # 验证相关的参数
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(
        valid_window, valid_size, replace=False)

    # 开始建图
    graph = tf.Graph()
    with graph.as_default():
        
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform(
                    [vocabulary_size, embedding_size], -1.0, 1.0))
            emb_vec = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=0.001))

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        with tf.name_scope('loss'):
            labels_2dim = tf.reshape(train_labels, [-1, 1])
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=weights,
                    biases=biases,
                    labels=labels_2dim,
                    inputs=emb_vec,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        with tf.name_scope('similarity'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normed_ebds = embeddings / norm #前两步做归一化处理
            vlid_embeddings = tf.nn.embedding_lookup(normed_ebds, valid_dataset)
            # 设valid_embedings选出来N个词，则shape 为 N * D，那个与(V * D).T 相乘后得到N * V
            # 每row里面存的是改词与其他各词的相似度
            similarity = tf.matmul(vlid_embeddings, normed_ebds, transpose_b=True)

        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        init.run()
        print 'Finish Variables Initialization...'
        avg_loss = 0
        for step in xrange(train_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            # 定义 metadata 变量， 记录训练运算时间和内存占用等信息。
            run_metadata = tf.RunMetadata()
            _, summary, loss_val = sess.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            avg_loss += loss_val
            # 每一步都将summary打出来
            writer.add_summary(summary, step)
            if step == (train_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if step % 2000 == 0 and step > 0:
                avg_loss /= 2000
                print('Average loss at step ', step, ': ', avg_loss)
                avg_loss = 0
            if step % 10000 == 0:
                sim = sess.run(similarity)
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    # 返回排序后的index， array 的 index
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print log_str
        # 结束循环后求一个归一化的embeding
        final_ebds = sess.run(normed_ebds)
        # 保存词汇表，行号为他们在字典里的index
        with open(os.path.join('log', 'metadata.tsv'), 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        #保存检查点
        saver.save(sess, os.path.join('log', 'model.checkpoint'))
        
        #可视化 embeding
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
        projector.visualize_embeddings(writer, config)
        writer.close()

        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_ebds[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join('tsne', 'tsne.png'))