## word2vec 实现

#### Corpus
`text8.small`
```sh
$ wc -w text8.small
output : 1706282
$ cat text8.small | tr ' ' '\n' | sort | uniq | wc -w 
output : 70985
```

`text8.small` 
```sh
$ wget http://mattmahoney.net/dc/text8.zip
$ wc -w text8
output : 17005207
$ cat text8 | tr ' ' '\n' | sort | uniq | wc -w 
output : 253854
```

#### How to run

```sh
python word2vec.py --log_path=small_log --corpus_file=text8.small --voc_size=30000
```
or
```
python word2vec.py
```

other params
```python
batch_size = FLAGS.batch_size
embedding_size = FLAGS.embedding_size
skip_window = FLAGS.skip_window
num_sampled = FLAGS.neg_sample_num
learning_rate = FLAGS.learning_rate

```

### some results
```
Nearest to which: that, this, also, but, it, wct, roshan, and,
Nearest to who: he, they, originally, which, chromatic, also, gibb, she,
Nearest to they: we, he, there, you, it, not, she, michelob,
Nearest to b: d, tentatively, pulau, j, six, UNK, arpa, rydberg,
Nearest to over: fairies, truetype, about, reductions, michelob, teachers, within, harmony,
Nearest to some: many, all, several, these, ursus, any, teschen, contagious,
Nearest to to: wct, iit, ursus, manure, will, circ, would, chalukyas,
Nearest to about: buckingham, four, over, hypothermia, adjustments, before, ursus, formalization,
Nearest to up: illyrians, dulles, him, incompressible, tg, subcode, off, memorized,
Nearest to it: he, this, there, she, wct, they, which, upanija,
Nearest to most: more, many, symbolics, relaxing, some, archie, fiorello, thaler,
Nearest to for: agouti, ursus, gigantopithecus, michelob, circ, braveheart, while, jati,
Nearest to after: before, during, when, xk, without, bayezid, later, since,
Nearest to state: dasyprocta, operatorname, vma, deregulation, wct, asymmetric, pontificia, crete,
Nearest to during: in, abet, by, when, after, iit, addington, dioxin,
Nearest to on: in, through, cebus, tano, numa, ursus, manure, against,
```

### reference

[1] Learn Word2Vec by implementing it in tensorflow https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac

[2] word2vec 中的数学原理详解 https://blog.csdn.net/itplus/article/details/37969519

[3] Word2Vec Tutorial Part 2 - Negative Sampling http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

[4] Vector Representations of Words https://www.tensorflow.org/tutorials/representation/word2vec?hl=en
