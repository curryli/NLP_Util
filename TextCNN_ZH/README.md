# 基于cnn的中文文本分类算法

## 简介
实现的一个简单的卷积神经网络，用于中文文本分类任务（此项目使用的数据集是中文垃圾邮件识别任务的数据集），数据集下载地址：[百度网盘](https://pan.baidu.com/s/1i4HaYTB)


在 https://github.com/clayandgithub/zh_cnn_text_classify 基础之上（源代码有点小问题），修改了一点，并加了一点说明 
主要改动点：
1、只保留中文之后，直接使用jieba直接分词.
1、原来代码中，如果预先使用word2vec，那么text_cnn.py需要改为
self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
否则出错