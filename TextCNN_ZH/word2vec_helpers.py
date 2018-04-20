# -*- coding: utf-8 -*-

'''
python word2vec_helpers.py input_file output_model_file output_vector_file
https://blog.csdn.net/szlcw1/article/details/52751314
https://blog.csdn.net/john_xyz/article/details/54706807
https://blog.csdn.net/xiaoquantouer/article/details/53583980
'''

# import modules & set up logging
import os
import sys
import logging
import multiprocessing
import time
import json
import numpy as np
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)

def embedding_sentences(sentences, embedding_size = 128, window = 5, min_count = 5, file_to_load = None, file_to_save = None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences, size = embedding_size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors


def generate_word2vec_files(input_file, output_model_file, output_vector_file, size = 128, window = 5, min_count = 5):
    start_time = time.time()

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file), size = size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    input_file, output_model_file, output_vector_file = sys.argv[1:4]

    generate_word2vec_files(input_file, output_model_file, output_vector_file) 

def mytest():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence'], ['third', 'sentence']], embedding_size = 5, min_count = 1)
    print(vectors)

    # [[array([-0.0295424, -0.06962696, 0.0141016, 0.04448513, 0.01840453], dtype=float32),
    #   array([0.02182164, -0.0301549, -0.06413952, -0.01989684, 0.00630156], dtype=float32)],
    #
    #  [array([0.0304694, -0.08486927, 0.01471626, -0.09844144, -0.04638988], dtype=float32),
    #   array([0.02182164, -0.0301549, -0.06413952, -0.01989684, 0.00630156], dtype=float32)],
    #
    #  [array([0.03501162, -0.0547439, 0.02174695, 0.03999822, 0.00576901], dtype=float32),
    #   array([0.02182164, -0.0301549, -0.06413952, -0.01989684, 0.00630156], dtype=float32)]]

    #print np.array(vectors).shape  #(3L, 2L, 5L)
    # 3句话  第一句 ['first', 'sentence']  第二句['second', 'sentence']  第三句 ['third', 'sentence']
    #每句话 2维

if __name__ == '__main__':
    mytest()
