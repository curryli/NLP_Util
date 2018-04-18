 # -*- coding: utf-8 -*-
import numpy as np
 
 
## 手动构建 ngram 数据集
#def create_ngram_set(input_list, ngram_value=2):
#    """
#    Extract a set of n-grams from a list of integers.
#    从一个整数列表中提取  n-gram 集合。
#    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
#    {(4, 9), (4, 1), (1, 4), (9, 4)}
#    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
#    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
#    """
#    return set(zip(*[input_list[i:] for i in range(ngram_value)]))
#
#
#def add_ngram(sequences, token_indice, ngram_range=2):
#    """
#    Augment the input list of list (sequences) by appending n-grams values.
#    增广输入列表中的每个序列，添加 n-gram 值
#    Example: adding bi-gram
#    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
#    >>> add_ngram(sequences, token_indice, ngram_range=2)
#    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
#    Example: adding tri-gram
#    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
#    >>> add_ngram(sequences, token_indice, ngram_range=3)
#    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
#    """
#    new_sequences = []
#    for input_list in sequences:
#        new_list = input_list[:]
#        for i in range(len(new_list) - ngram_range + 1):
#            for ngram_value in range(2, ngram_range + 1):
#                ngram = tuple(new_list[i:i + ngram_value])
#                if ngram in token_indice:
#                    new_list.append(token_indice[ngram])
#        new_sequences.append(new_list)
#
#    return new_sequences
 
    
from nltk.util import ngrams  
a = ['我','爱','北京','人','爱','北京']  
b = ngrams(a,3)  
for i in b:  
     print i  
     
     
  #实现了标记和计数   
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)


corpus = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
 ]
X = vectorizer.fit_transform(corpus)

#每个词代表一列
print vectorizer.get_feature_names()

#print X 
#这样每一句话出现每个词的统计放在一行中  
print X.toarray()   

#this是第8列
print vectorizer.vocabulary_.get('this')

#注意在前面的语料中，第一个和最后一个文档的词完全相同因此被编码为等价的向量。
#为了保留一些局部顺序信息 我们可以在抽取词的1-grams（词本身）之外，再抽取2-grams：

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
#analyze = bigram_vectorizer.build_analyzer()
#既有1gram 又有2gram， 如果只想要2gram，可以取差集？
#print analyze('Bi-grams are cool!')

X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print X_2


#在较低的文本语料库中，一些词非常常见（例如，英文中的“the”，“a”，“is”），因此很少带有文档实际内容的有用信息。如果我们将单纯的计数数据直接喂给分类器，那些频繁出现的词会掩盖那些很少出现但是更有意义的词的频率。
#为了重新计算特征的计数权重，以便转化为适合分类器使用的浮点值，通常都会进行tf-idf转换。

#http://www.sohu.com/a/198378499_642762

#CountVectorizer    TfidfVectorizer 都会在内存中生成词汇表 （所以get_feature_names()可以获得名称）       HashingVectorizer(不生成词汇表，快 但不可逆，可以在管道中添加TfidfTransformer)
              