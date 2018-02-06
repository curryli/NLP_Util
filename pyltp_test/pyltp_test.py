# -*- coding: utf-8 -*-
#安装方法见   http://blog.sina.com.cn/s/blog_735f29100102x6vp.html
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





#分句，也就是将一片文本分割为独立的句子
def sentence_splitter(sentence='你好，你觉得这个例子从哪里来的？当然还是直接复制官方文档，然后改了下这里得到的。'):
    sents = SentenceSplitter.split(sentence)  # 分句
    print '\n'.join(sents)



def role_label(words, postags, netags, arcs):
    labeller = SementicRoleLabeller() # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    roles = labeller.label(words, postags, netags, arcs)  # 语义角色标注
    for role in roles:
        print role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
    labeller.release()  # 释放模型





if __name__ == '__main__':
    #分句子
    sentence_splitter()

    ##############################################分词####################################################
    #简单分词
    #words = segmentor.segment('元芳你怎么看') 的返回值类型是native的VectorOfString类型，可以使用list转换成Python的列表类型，例如
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment('元芳你怎么看')  # 分词
    print ' '.join(list(words))
    segmentor.release()  # 释放模型

    #使用分词外部词典
    #pyltp 分词支持用户使用自定义词典。分词外部词典本身是一个文本文件（plain text），每行指定一个词，编码同样须为 UTF-8，样例如下所示
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path, './my_dict/split_test1.txt')  # 加载模型，('模型地址, '用户字典')  这样既结合了训练好的cws.model，又结合了自定义的词典
    words = segmentor.segment('亚硝酸盐是一种化学物质')
    print ' '.join(words)
    segmentor.release()

    #使用个性化分词模型  暂时用不到，分词甚至可以直接使用JIEBA，效果可能更好   这边就不说了

    #####################################################词性标注###################################################
    #参数 words 是分词模块的返回值，也支持Python原生的list类型，例如
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    words = ['元芳', '你', '怎么', '看']  # 分词结果
    postags = postagger.postag(words)  # 词性标注
    print ' '.join(postags)
    postagger.release()  # 释放模型

    #pyltp 词性标注同样支持用户的外部词典。
    #第一列指定单词，第二列之后指定该词的候选词性（可以有多项，每一项占一列），列与列之间用空格区分。
    postagger = Postagger()  # 初始化实例
    postagger.load_with_lexicon(pos_model_path, "./my_dict/pos_test1.txt")
    words = ['雷人', '的', '杜甫']  # 分词结果
    postags = postagger.postag(words)  # 词性标注
    print ' '.join(postags)
    postagger.release()  # 释放模型

    #####################################################命名实体识别###################################################
    #其中，words 和 postags 分别为分词和词性标注的结果。同样支持Python原生的list类型。
    #LTP 采用 BIESO 标注体系。B 表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体，O表示不构成命名实体。
    # LTP 提供的命名实体类型为:人名（Nh）、地名（Ns）、机构名（Ni）。
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    words = ['元芳', '你', '怎么', '看']
    postags = ['nh', 'r', 'r', 'v']
    netags = recognizer.recognize(words, postags)  # 命名实体识别

    print ' '.join(netags)
    recognizer.release()  # 释放模型

    ##连起来的用法
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment('美国微软公司的总部设在哪里')

    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(words)

    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(words, postags)

    print ' '.join(netags)  #B-Ni I-Ni E-Ni O O O O O    美国微软公司整体是一个命名实体，美国是实体开始词， 微软是实体中间词，公司是实体结束词
    recognizer.release()

    ############################################依存句法分析################################################
    #其中，words 和 postags 分别为分词和词性标注的结果。同样支持Python原生的list类型。
    #
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    words = ['元芳', '你', '怎么', '看']
    postags = ['nh', 'r', 'r', 'v']
    arcs = parser.parse(words, postags)  # 句法分析

    print " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
    parser.release()  # 释放模型
    #4:SBV 4:SBV 4:ADV 0:HED
    # 可以到   https://www.ltp-cloud.com/demo/  输入    元芳你怎么看  点分析看结果  结果图片保存在my_dict下面
    #解释：ROOT节点的索引是0，第一个词开始的索引依次为  元芳1、你2、怎么3、看4 …
    #可以发现   看 是核心词  HED


    ###########################################语义角色标注###############################
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment('元芳你怎么看')

    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(words)

    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    netags = recognizer.recognize(words, postags)
    recognizer.release()

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words, postags)  # 句法分析


    # j角色标注
    roles = role_label(words, postags, netags, arcs)


    #3  A0:(0, 0)  A0:(1, 1)  ADV:(2, 2)
    #分析，这边只输出一行，表示只存在一组语义角色。 其谓词索引为3，即“看”。这个谓词有三个语义角色，范围分别是(0,0)即“元芳”，(1,1)即“你”，(2,2)即“怎么”，类型分别是A0、A0、ADV。
    #核心的语义角色为 A0-A5 六种  具体表述见https://www.ltp-cloud.com/intro/
    #(0,0) 表示范围   如果上来是  美国微软公司  ，这个范围可能就是(0,2)表示这个核心语义角色由0~2组成


    #http://pyltp.readthedocs.io/zh_CN/develop/api.html
    # http://blog.csdn.net/MebiuW/article/details/52496920


