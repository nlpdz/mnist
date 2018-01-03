# -*- coding:utf8 -*-
from __future__ import print_function
import readimg
import common
import os,time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

sess, y_conv, x, y_, keep_prob, train_step, correct_prediction, accuracy = common.preduce()

saver = tf.train.Saver()
save_path = './tmp/model.ckpt'
try:
  # arg:获取最近一次保存的变量文件名称
  module_file = tf.train.latest_checkpoint('./tmp')
  print(module_file)
  saver.restore(sess, module_file)
  print("Model restored.")
except:
  pass


def pltinit():
    fig = plt.gcf()
    fig.set_size_inches(0.28, 0.28)
    plt.xticks([])
    plt.yticks([])
    # 关闭坐标轴
    plt.axis('off')

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# batch = mnist.train.next_batch(1)
path = './img/2'
# path = './image/88.png'
items = os.listdir(path)
for item in items:

    img =readimg.load_img_to_center(os.path.join(path,item))
    # img = batch[0].reshape(28, 28)*255
    pltinit()
    plt.imshow(img, cmap='Greys')
    # plt.savefig('./img/t.png')
    plt.show()

    batchimg = img.reshape(1,28*28)

    predict = y_conv.eval(feed_dict={
        x: batchimg, keep_prob: 1.0})
    print('predict is ')
    t = predict.tolist()
    print(t)
    num = 0
    predi = t[0].index(max(t[0]))
    print('预测的结果是%d' % predi)
    # for i in t[0]:
    #     if i > 0.1:
    #         print('预测的结果是%d' % num)
    #     num = num + 1
    time.sleep(1)
