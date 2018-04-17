#! /usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers



# Data preprocess
# =======================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_positive_negative_data_files("./data/ham_100.utf8", "./data/spam_100.utf8")
#print len(x_text)





# # Build vocabulary   如果用这种方法，后面训练网络的时候要加embedding层
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))


# Get embedding vector 如果用这种方法，后面训练网络的时候应该不要加embedding层
sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = 128, file_to_save = os.path.join("./", 'trained_word2vec.model')))
print("x.shape = {}".format(x.shape))  # x.shape = (200L, 282L, 128L)   200行  每句话被 pad到 282L个单词   每个单词embedding_size = 128
print("y.shape = {}".format(y.shape))  # y.shape = (200L, 2L)





