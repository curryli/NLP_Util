# -*- coding: utf-8 -*-
import jieba
import re

#stopwords = []
#
## jieba.load_userdict('userdict.txt')
## 创建停用词list
#def stopwordslist(filepath):
#    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
#    return stopwords
#
#
## 对句子进行分词
#def seg_sentence(sentence):
#    sentence_seged = jieba.cut(sentence.strip())
#   
#    outstr = ''
#    for word in sentence_seged:
#        if word not in stopwords:
#            if word != '\t':
#                outstr += word
#                outstr += " "
#    return outstr
# 
#
#
#stopwords = stopwordslist('stopwords_zh.txt')  # 这里加载停用词的路径
# 
#inputs = open('input.txt', 'r')
#outputs = open('output.txt', 'w')
#for line in inputs:
#    line_seg = seg_sentence(line)  # 这里的返回值是字符串
#    outputs.write(line_seg + '\n')
#outputs.close()
#inputs.close()


#只提取中文
pattern =re.compile(u"[\u4e00-\u9fa5]+")

#只提取中文 字母 和数字
#pattern =re.compile(u"[\u4e00-\u9fa5a-zA-Z0-9]+")
 
inputs = open('input.txt', 'r')

print "start"
for line in inputs:
    group_zh = re.findall(pattern,line.decode("utf8"))
    line_zh = " ".join(group_zh)
    line_seg = jieba.cut(line_zh.strip())
    print ",".join(line_seg)
 
    
inputs.close()





