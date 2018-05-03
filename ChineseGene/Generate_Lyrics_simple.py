# coding: utf-8

import helper
import re

# 读入数据

dir = './data/JayLyrics.txt'
text = helper.load_text(dir)

lines_of_text = text.split('\n')

# 去掉每行首尾空格
lines_of_text = [lines.strip() for lines in lines_of_text]

# 到这里数据就处理完了。再看看有多少行数据。
print(len(lines_of_text))


# 因为模型只认识数字，不认识中文，所以将文字对应到数字，分别创建文字对应数字和数字对应文字的两个字典

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))

    return vocab_to_int, int_to_vocab


# 创建一个符号查询表，把逗号，句号等符号与一个标志一一对应，用于将『我。』和『我』这样的类似情况区分开来，排除标点符号的影响。

# In[25]:

def token_lookup():

    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))


# 预处理一下数据，并保存到磁盘
helper.preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)


# 读取我们需要的数据
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# 检查改一下当前Tensorflow的版本以及是否有GPU可以使用


import problem_unittests as tests
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



# 首先设置一下超参。

# 训练循环次数
num_epochs = 100

# batch大小
batch_size = 128

# lstm层中包含的unit个数
rnn_size = 256

# embedding layer的大小   这里设置的要和rnn_size一样
embed_dim = 256

# 训练步长
seq_length = 32

# 学习率
learning_rate = 0.01

# 每多少步打印一次训练信息
show_every_n_batches = 8

# 保存session状态的位置
save_dir = './save'


# 创建输入，目标以及学习率的placeholder
def get_inputs():
    
    # inputs和targets的类型都是整数的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs, targets, learning_rate


# 创建rnn cell，使用lstm cell，并创建相应层数的lstm层，应用dropout，以及初始化lstm层状态。

def get_init_cell(batch_size, rnn_size):
    # lstm层数
    num_layers = 3
        
    # dropout时的保留概率
    keep_prob = 0.8
    
    # 创建包含rnn_size个神经元的lstm cell
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # 使用dropout机制防止overfitting等
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
    # 创建2层lstm层
    cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])
    
    # 初始化状态为0.0
    init_state = cell.zero_state(batch_size, tf.float32)
    
    # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state
    init_state = tf.identity(init_state, name='init_state')

    return cell, init_state


# 创建embedding layer，提升效率

# In[32]:

def get_embed(input_data, vocab_size, embed_dim):
    
    # 先根据文字数量和embedding layer的size创建tensorflow variable
    embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_dim], stddev=0.1), 
                            dtype=tf.float32, name="embedding")
    
    # 让tensorflow帮我们创建lookup table
    return tf.nn.embedding_lookup(embedding, input_data, name="embed_data")


# 创建rnn节点，使用dynamic_rnn方法计算出output和final_state


def build_rnn(cell, inputs):
    
    '''
    cell就是上面get_init_cell创建的cell
    '''
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    
    # 同样给final_state一个名字，后面要重新获取缓存
    final_state = tf.identity(final_state, name="final_state")
    
    return outputs, final_state


# 用上面定义的方法创建rnn网络，并接入最后一层fully_connected layer计算rnn的logits

# In[34]:

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    
    # 创建embedding layer
    embed = get_embed(input_data, vocab_size, embed_dim)
    
    # 计算outputs 和 final_state
    outputs, final_state = build_rnn(cell, embed)
    
    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())
    
    # logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    
    return logits, final_state


# 那么大的数据量不可能一次性都塞到模型里训练，所以用get_batches方法一次使用一部分数据来训练

# In[35]:

def get_batches(int_text, batch_size, seq_length):

    # 计算有多少个batch可以创建
    n_batches = (len(int_text) // (batch_size * seq_length))

    # 计算每一步的原始数据，和位移一位之后的数据
    batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
    batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    # 将位移之后的数据的最后一位，设置成原始数据的第一位，相当于在做循环
    batch_shifted[-1] = batch_origin[0]

    batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

    batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))

    return batches


# 创建整个RNN网络模型

# 导入seq2seq，下面会用他计算loss
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    # 文字总量
    vocab_size = len(int_to_vocab)

    # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
    input_text, targets, lr = get_inputs()
    
    # 输入数据的shape
    input_data_shape = tf.shape(input_text)
    
    # 创建rnn的cell和初始状态节点，rnn的cell已经包含了lstm，dropout
    # 这里的rnn_size表示每个lstm cell中包含了多少的神经元
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    
    # 创建计算loss和finalstate的节点
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # 使用softmax计算最后的预测概率
    probs = tf.nn.softmax(logits, name='probs')


    # 计算loss
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # 使用Adam提督下降
    optimizer = tf.train.AdamOptimizer(lr)

    # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# 开始训练模型

# In[39]:

# 获得训练用的所有batch
batches = get_batches(int_text, batch_size, seq_length)

# 打开session开始训练，将上面创建的graph对象传递给session
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # 打印训练信息
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# 将使用到的变量保存起来，以便下次直接读取。

# In[40]:

helper.save_params((seq_length, save_dir))


# 下次使用训练好的模型，从这里开始就好

# In[41]:

import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# 要使用保存的模型，我们要讲保存下来的变量（tensor）通过指定的name获取到

# In[42]:

def get_tensors(loaded_graph):
   
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    
    initial_state = loaded_graph.get_tensor_by_name("init_state:0")
    
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    
    probs = loaded_graph.get_tensor_by_name("probs:0")
    
    return inputs, initial_state, final_state, probs


# 使用训练好的模型来生成自己的小说

# 生成文本的长度
gen_length = 500

# 文章开头的字，指定一个即可，这个字必须是在训练词汇列表中的
prime_word = '我'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存过的session
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # 通过名称获取缓存的tensor
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # 准备开始生成文本
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, feed_dict = {input_text: np.array([[1]])})

    # 开始生成文本
    #print(int_to_vocab)
    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]  #最近的seq_length个字
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            feed_dict= {input_text: dyn_input, initial_state: prev_state})

        # print(probabilities[0].shape)
        # print(probabilities[0][-1].shape)
        # print(np.argmax(probabilities[0][-1]))
        pred_word =  int_to_vocab[np.argmax(probabilities[0][-1])]
        gen_sentences.append(pred_word)


    # 将标点符号还原
    novel = ''.join(gen_sentences)
    # for key, token in token_dict.items():
    #     ending = ' ' if key in ['\n', '（', '“'] else ''
    #     novel = novel.replace(token.lower(), key)

    print(novel)

    fh = open("gened.txt", 'w')
    fh.write(novel)
    fh.close()


# In[ ]:



