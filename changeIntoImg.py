# -*- coding: utf8 -*-
''' 
将mnist中的数据转为图片，存储在当前目录下的  image  文件夹下

'''
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    print(os.path.abspath(labels_path))
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def init():
    fig = plt.gcf()
    fig.set_size_inches(0.28, 0.28)
    plt.xticks([])
    plt.yticks([])
    # 关闭坐标轴
    plt.axis('off')


X_train, y_train = load_mnist('./MNIST_data/')

path = './image'
if not os.path.exists(path):
    os.mkdirs(path)

for i in range(len(X_train)):
    init()
    img = X_train[i].reshape(28, 28)
    plt.imshow(img, cmap='Greys')
    plt.savefig(os.path.join(path, '%d.png' % i))
    plt.cla()
