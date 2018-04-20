# encoding: UTF-8

import numpy as np
import re
import itertools
from collections import Counter
import os
import word2vec_helpers
import time
import pickle

import os
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SementicRoleLabeller


LTP_DATA_DIR = 'D:\Libs_for_All\NLP_Libs\ltp_data_v3.3.1'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。

seg = Segmentor()
seg.load(cws_model_path)



def load_positive_negative_data_files(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = read_and_clean_zh_file(positive_data_file)
    negative_examples = read_and_clean_zh_file(negative_data_file)
    # Combine data
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def padding_sentences(sentences, padding_token, padding_sentence_length = None):
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)



def mytest():
    print("Test")
    x_text,y = load_positive_negative_data_files("./data/ham_100.utf8", "./data/spam_100.utf8")
    print(x_text)
    print(y)



# #分句，也就是将一片文本分割为独立的句子
# def sentence_splitter(sentence='你好，你觉得这个例子从哪里来的？'):
#     sents = SentenceSplitter.split(sentence)  # 分句
#     print '\n'.join(sents)


#ltp的分词模型
def segmentor(sentence="你好，广东外语外贸大学欢迎你。"):
    words = seg.segment(sentence)
    words_list = list(words)
    return words_list

def read_and_clean_zh_file(input_file, output_cleaned_file = None):
    lines = []
    with open(input_file, 'r') as fin:
        for line in fin:
            line = segmentor(line)
            lines.append(line)
    return lines

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f) 

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict


if __name__ == '__main__':
    mytest()
