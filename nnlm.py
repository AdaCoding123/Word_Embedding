#encoding:utf-8
import sys
from input_data import *
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import time
import math
from tensorflow.python.platform import gfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',help='data directory containing input.txt')  #训练数据存储位置
    parser.add_argument('--batch_size', type=int, default=120,help='minibatch size')                 
    parser.add_argument('--win_size', type=int, default=5,help='context sequence length')    #输入的单词个数     
    parser.add_argument('--hidden_num', type=int, default=64,help='number of hidden layers')       
    parser.add_argument('--word_dim', type=int, default=50,help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=10,help='number of epochs')               
    parser.add_argument('--grad_clip', type=float, default=10.,help='clip gradients at this value')  

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size
    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #定义训练数据，原始文本,one hot形式的词？？？
        input_data = tf.placeholder(tf.int32, [args.batch_size, args.win_size])
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])
        #模型参数,look-up表
        with tf.variable_scope('nnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))#vocab_size
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        with tf.variable_scope('nnlm' + 'weight'):
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim + 1, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num + 1, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.hidden_num)))


        #TODO，构造计算图
        def infer_output(input_data):
            # step 1: hidden = tanh(x * H + d)
            # step 2: outputs = softmax(x * W + hiddens * U + b) 
            H = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim,args.hidden_num],stddev=1.0 / math.sqrt(args.hidden_num)))   
            d = tf.Variable(tf.truncated_normal([1,args.hidden_num],stddev=1.0 / math.sqrt(args.hidden_num)))
            U = tf.Variable(tf.truncated_normal([args.hidden_num,args.vocab_size],stddev=1.0 / math.sqrt(args.hidden_num)))
            
            W =tf.Variable(tf.truncated_normal([args.win_size * args.word_dim,args.vocab_size],stddev=1.0 / math.sqrt(args.hidden_num)))
            b = tf.Variable(tf.truncated_normal([1,args.vocab_size ],stddev=1.0 / math.sqrt(args.hidden_num)))
           
            #得到e低维稠密向量
            input_data_emb=tf.nn.embedding_lookup(embeddings,input_data) 
            #得到拼接的词向量
            input_data_emb=tf.reshape(input_data_emb,[-1,args.win_size*args.word_dim])
                      
            hidden = tf.tanh(tf.matmul(input_data_emb,H) + d)
            outputs = tf.nn.softmax(tf.matmul(input_data_emb,W)+tf.matmul(hidden,U)+b)
            return outputs

        outputs = infer_output(input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0)

        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))
        optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        #输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm

    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss,  _ = sess.run([loss, optimizer], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))

             # 保存词向量至nnlm_word_embeddings.npy文件
            np.save('nnlm_word_embeddings.en', normalized_embeddings.eval())
        

if __name__ == '__main__':
    main()
