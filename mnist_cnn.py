# -*- coding:utf8 -*-
from __future__ import print_function
import sys
import common
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

sess,y_conv,x ,y_,keep_prob,train_step,correct_prediction,accuracy = common.preduce()

# 开始
flags = 'train'
# flags = 'assessment'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

itera_count = 0

saver = tf.train.Saver()
save_path = './tmp/model.ckpt'
iterfiles = './tmp/iter.txt'
try:
    with open(iterfiles) as iterfile:
        itera_count = iterfile.read()
        if itera_count == '':
            itera_count = 0
        else:
            itera_count = int(itera_count)
    # arg:获取最近一次保存的变量文件名称
    module_file = tf.train.latest_checkpoint('./tmp')
    print(module_file)
    saver.restore(sess, module_file)
    print("Model restored.")
except:
    pass

if flags == 'train':
    # 训练
    print('训练开始')
    for i in range(itera_count, 1000000 + itera_count, 1):
        print(i, end='\t')
        batch = mnist.train.next_batch(50)
        if i % 100 == 0 and i != itera_count:
            # 保存变量
            with open(iterfiles, 'w') as iterfile:
                itera_count = iterfile.write(str(i))
                savepath = saver.save(sess, save_path, global_step=i)
                print("Model saved in file: %s" % savepath)

            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print('>>step %d, training accuracy %.4f' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
elif flags == 'assessment':
    # 评估
    print('评估开始')
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
