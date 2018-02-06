# -*- coding: utf-8 -*-

#from __future__ import absolute_import, unicode_literals   这个是为了兼容python3，在python2里面运行会出错
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
import os

from pyltp_util import segmentor,posttagger

class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws

class my_TextRank():

    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
        self.span = 5


    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        g = UndirectWeightedGraph()
        cm = defaultdict(int)

        wordlist = segmentor(sentence)
        taglist = posttagger(wordlist)

        words = zip(range(0,len(wordlist)),zip(wordlist, taglist))
        words = [w for w in words if w[1][1] in allowPOS]

        for i, wp in enumerate(words):
            for j in xrange(i + 1, i + self.span):
                if j >= len(words):  #快到末尾的时候，加了span可能溢出，忽略
                    break
                else:
                    cm[(wp[1], words[j][1])] += 1


        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

    extract_tags = textrank
