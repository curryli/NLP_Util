# coding=utf-8
__author__ = 'gu'

from pyltp import Segmentor,Postagger,SentenceSplitter,Parser
import os

class RoleRelation:
    """
    获取人物关系的代码类
    """

    LTP_DATA_DIR = 'D:\Libs_for_All\NLP_Libs\ltp_data_v3.3.1'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。

    book_root_path = "../mybooks/Book"
    relation_root_path = "../mybooks/Relation"
    nh_relation_root_path = "../mybooks/NHRelation"

    seg = Segmentor()
    seg.load(cws_model_path)
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型

    def getAllPath(self):
        txtpathlist = []
        dirlist = os.listdir(self.book_root_path)
        for dirname in dirlist:
            print dirname
            path = self.book_root_path + "/"+dirname
            txtlist = os.listdir(path)
            for txtname in txtlist:
                #print txtname
                txtpath = path + "/"+txtname
                txtpathlist.append(txtpath)
        #print len(txtpathlist)
        return txtpathlist

    def segmentor(self, sentence="你好，广东外语外贸大学欢迎你。"):
        """
        ltp的分词模型
        :return:
        :param sentence:
        """

        words = self.seg.segment(sentence)
        words_list = list(words)
        # for word in words_list:
        #     print word
        # self.seg.release()
        return words_list

    def posttagger(self,words):

        postags = self.postagger.postag(words)  # 词性标注
        # for word,tag in zip(words,postags):
        #     print word+'/' + tag
        # postagger.release()  # 释放模型
        return list(postags)

    def myparse(self, words, postags):
        arcs = self.parser.parse(words, postags)  # 句法分析
        # arc.head 表示依存弧的父节点词的索引，arc.relation 表示依存弧的关系
        # for w in words:
        #     print w,
        # print
        # for arc in arcs:
        #     print arc.head,":",arc.relation,
        # print
        SOVS = []
        for index_subject in range(len(arcs)):
            if arcs[index_subject].relation == "SBV":
                subjects = [words[index_subject]]  # 主语列表
                predicate = words[arcs[index_subject].head-1] # 谓语
                binyus = [] # 宾语列表

                index_bin = -1
                for index_coo in range(len(arcs)):
                    if arcs[index_coo].relation == "COO" and arcs[index_coo].head == index_subject+1:
                        subjects.append(words[index_coo])  # 并列主语
                    elif arcs[index_coo].relation == "VOB" and arcs[index_coo].head == arcs[index_subject].head:
                        index_bin = index_coo
                        binyus.append(words[index_coo])
                    elif index_bin > 0 and arcs[index_coo].relation == "COO" and arcs[index_coo].head == index_bin + 1:
                        binyus.append(words[index_coo])

                for subj in subjects:
                    if len(binyus) > 0:
                        for binyu in binyus:
                            #print subj, predicate, binyu
                            SOVS.append((subj, predicate, binyu))
                    else:
                        #print subj,predicate
                        SOVS.append((subj,predicate))
        #print
        return SOVS

    def getRoleReation(self,path="../mybooks/Book/西游记/1.txt"):
        rf = open(path,"r")
        lines = rf.readlines()
        rf.close()
        SOVS = []
        for line in lines:
            line = line.replace("\n","").replace("\r","")
            if line != "":
                sents = SentenceSplitter.split(line)
                for sent in sents:
                    #print sent
                    words = self.segmentor(sent)
                    postags = self.posttagger(words)
                    SOVS += self.myparse(words,postags)
        return SOVS

    def getAllRelation(self):
        darlist = os.listdir(self.book_root_path)
        for dirname in darlist:
            print dirname
            path = self.book_root_path + "/"+dirname
            txtlist = os.listdir(path)
            for txtname in txtlist:
                txtpath = path + "/"+txtname
                SOVS = self.getRoleReation(txtpath)
                if len(SOVS) > 0:
                    print "=====start========"
                    for sov in SOVS:
                        for s in sov:
                            print s,
                        print
                    print "======end======"
                    self.writeRelation(SOVS,txtname,self.relation_root_path+"/"+dirname)

    def writeRelation(self,SOVS,filename,path):
        wf = open(path+"/"+filename,"w")
        for sov in SOVS:
            for s in sov:
                wf.write(s+" ")
            wf.write("\n")
        wf.close()



    def getNH(self,path=book_root_path+"/西游记/1.txt"):
        rf = open(path,"r")
        lines = rf.readlines()
        rf.close()
        names = []
        for line in lines:
            line = line.replace("\n","").replace("\r","")
            if line != "":
                sents = SentenceSplitter.split(line)
                for sent in sents:
                    #print sent
                    words = self.segmentor(sent)
                    postags = self.posttagger(words)
                    for word,tag in zip(words,postags):
                        if tag == "nh":
                            #print word+'/' + tag
                            names.append(word)
        return list(set(names))

    #这个用  NamedEntityRecognizer 进行命名实体识别 找出人名存下来 是不是更好
    def getAllNH(self):
        darlist = os.listdir(self.book_root_path)
        names_book = []
        for dirname in darlist:
            #print dirname
            path = self.book_root_path + "/"+dirname
            txtlist = os.listdir(path)
            for txtname in txtlist:
                txtpath = path + "/"+txtname
                names_chapter = self.getNH(txtpath)
                names_book += names_chapter
        self.savaNames(list(set(names_book)))

    def savaNames(self,names,path=nh_relation_root_path+"/allnames.txt"):
        wf = open(path,"w")
        for name in names:
            wf.write(name+"\n")
        wf.close()


    def loadNameDict(self,path=nh_relation_root_path+"/allnames.txt"):
        rf = open(path,"r")
        name_dict = {}
        lines = rf.readlines()
        print len(lines)
        for line in lines:
            line = line.replace("\n","").replace("\r","")
            name_dict[line] = 1
        return name_dict


    def fliterRelation(self,names_dict,path=relation_root_path + "/三国演义白话文/1.txt"):
        rf = open(path,"r")
        lines = rf.readlines()
        print len(lines)
        nhline = []
        rf.close()
        for line in lines:
            line = line.replace("\n","").replace("\r","")
            spline = line.split(" ")
            if len(spline) > 0:
                if names_dict.__contains__(spline[0]):
                    nhline.append(line)
                    print line
        return nhline


    def fliterAllRelation(self):
        """
        过滤掉主谓宾中非人名的主语
        :return:
        """
        names_dict = self.loadNameDict()
        darlist = os.listdir(self.relation_root_path)
        for dirname in darlist:
            print dirname
            path = self.relation_root_path + "/"+dirname
            txtlist = os.listdir(path)
            for txtname in txtlist:
                print txtname
                nhline_chapter = self.fliterRelation(names_dict,path+"/"+txtname)
                self.savaNames(nhline_chapter,self.nh_relation_root_path+"/"+dirname+"/"+txtname) # savaNHRelation

if __name__ == '__main__':
    rr = RoleRelation()
    # #先运行rr.getAllNH，获得book_root_path下文章的所有人名，存到\mybooks\NHRelation\allnames.txt  供后面调用
    # rr.getAllNH()
    #
    # #保存所有关系
    # rr.getAllRelation()

    #过滤掉主语非人名的关系
    rr.fliterAllRelation()



