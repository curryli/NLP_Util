# -*- coding: utf-8 -*-
import nltk

__author__ = 'hello'

from pyltp import Segmentor
from pyltp import Postagger,SentenceSplitter
import os


class Books:
    """
    获取每一章节的主要人物 和整本书的主要人物
    """
    LTP_DATA_DIR = 'D:\Libs_for_All\NLP_Libs\ltp_data_v3.3.1'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。

    book_root_path = "../mybooks/Book/"
    mainrole_root_path = "../mybooks/MainRole/"
    mainlo_root_path = "../mybooks/MainLocaltion/"

    seg = Segmentor()
    seg.load(cws_model_path)
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型


    def readBookLines(self, path):
        rf = open(path, "r")
        lines = rf.readlines()
        rf.close()
        return lines

    def writeTxt(self, path, namelist):
        wf = open(path, "w")
        for name,times,freq in namelist:
            wf.write(str(name)+ " "+str(times)+" "+str(freq) + "\n")
        wf.close()

    #ltp的分词模型
    def segmentor(self, sentence="你好，广东外语外贸大学欢迎你。"):
        words = self.seg.segment(sentence)
        words_list = list(words)
        return words_list

    #ltp的词性标注,获取地理信息nl或者ns
    def posttagerNLNS(self,word_list):
        """
        词性标注之后，只返回 方位指示词nl 和 地理名称ns类型的词
        nl  location noun   城郊
        ns  geographical name   北京
        一个汉字占两个字节，所以len(word) > 3表示至少两个汉字
        """
        postags = self.postagger.postag(word_list)  # 词性标注
        localtion_list = []
        for word, tag in zip(word_list, postags):
            if (tag == "nl" or tag == "ns") and len(word) > 3:
                #print word + '/' + tag
                localtion_list.append(word)
        # postagger.release()  # 释放模型
        return localtion_list


    #ltp的词性标注,获取人名nh
    def posttaggerNH(self, word_list):
        postags = self.postagger.postag(word_list)  # 词性标注
        name_list = []
        for word, tag in zip(word_list, postags):
            if tag == "nh" and len(word) > 3:
                #print word + '/' + tag
                name_list.append(word)
        # postagger.release()  # 释放模型
        return list(postags), name_list

    #item times freq == itf
    def getTopTen(self, namelist):
        resultitf = []
        resultname = []
        top10Name = []
        chapter_fdist = nltk.FreqDist(namelist)
        top_name_list = sorted(chapter_fdist.iteritems(), key=lambda x: x[1], reverse=True)
        for name, num in top_name_list[0:10]:
            tmplist = [name] * num
            top10Name+=tmplist
            resultname.append(name)
        chapter_fdist_ten = nltk.FreqDist(top10Name)
        for name1, num1 in sorted(chapter_fdist_ten.iteritems(), key=lambda x: x[1], reverse=True):
            #print name1,num1,round(float(chapter_fdist_ten.freq(name1)), 2)
            resultitf.append((name1,num1,round(float(chapter_fdist_ten.freq(name1)), 2)))
        return resultitf,resultname

    def mainLocaltion(self,dirName="西游记白话文"):
        txtlist = os.listdir(self.book_root_path+dirName)
        lo_list_book = []
        for txt in txtlist:
            lo_list_chapter = []
            print txt
            lines = self.readBookLines(self.book_root_path+dirName + "/" + txt)
            for line in lines:
                if line != "":
                    sents = SentenceSplitter.split(line)
                    for sent in sents:
                        words_line = self.segmentor(sent)
                        lo_list_line = self.posttagerNLNS(words_line)
                        lo_list_chapter += lo_list_line
            # 统计每一章节top 10
            top_itf_chapter,top_lo_chapter = self.getTopTen(lo_list_chapter)
            lo_list_book += top_lo_chapter
            self.writeTxt(self.mainlo_root_path+dirName + "/" + txt, top_itf_chapter)
            print txt+"本章节top 10----------------------"
            for cloname,clotimes,clofreq in top_itf_chapter:
                print cloname,clotimes,clofreq
        # 统计整本书 top 10
        top_loitf_book,top_lo_book = self.getTopTen(lo_list_book)
        self.writeTxt(self.mainlo_root_path+dirName + "/AllChapter.txt", top_loitf_book)
        print "整本书 top 10----------------------"
        for bloname,blotimes,blofreq in top_loitf_book:
            print bloname,blotimes,blofreq

    def mainName(self, dirName):
        txtlist = os.listdir(self.book_root_path+dirName)
        name_list_book = []
        for txt in txtlist:
            name_list_chapter = []
            print txt
            lines = self.readBookLines(self.book_root_path+dirName + "/" + txt)
            for line in lines:
                if line != "":
                    sents = SentenceSplitter.split(line)
                    for sent in sents:
                        words_line = self.segmentor(sent)
                        postags_line, name_list_line = self.posttaggerNH(words_line)
                        name_list_chapter += name_list_line
            # 统计每一章节top 10
            top_itf_chapter,top_name_chapter = self.getTopTen(name_list_chapter)  # [(name,times,freq),()]
            name_list_book += top_name_chapter
            self.writeTxt(self.mainrole_root_path+dirName + "/" + txt, top_itf_chapter)
            print txt+"本章节top 10----------------------"
            for cname,ctimes,cfreq in top_itf_chapter:
                print cname,ctimes,cfreq
        # 统计整本书 top 10
        top_itf_book,top_name_book = self.getTopTen(name_list_book)
        self.writeTxt(self.mainrole_root_path+dirName + "/AllChapter.txt", top_itf_book)
        print "整本书 top 10----------------------"
        for bname,btimes,bfreq in top_itf_book:
            print bname,btimes,bfreq

    def getAllMainName(self):
        dirNames = os.listdir(self.book_root_path)
        for dirname in dirNames:
            print dirname
            self.mainName(dirname)

    def getAllMainLo(self):
        dirNames = os.listdir(self.book_root_path)
        for dirname in dirNames:
            print dirname
            self.mainLocaltion(dirname)


if __name__ == '__main__':
    book = Books()
    book.getAllMainLo()
    book.getAllMainName()
