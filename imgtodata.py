# -*- coding: utf8 -*-
import os
from PIL import Image,ImageEnhance
import numpy as np
import time
import matplotlib.pyplot as plt
import readimg
import tensorflow.examples.tutorials.mnist.input_data as input_data

def pltinit():
    fig = plt.gcf()
    fig.set_size_inches(0.28, 0.28)
    plt.xticks([])
    plt.yticks([])
    # 关闭坐标轴
    plt.axis('off')


def numtohot(num):
    """ 将num转换成10个二进制的数 例如num= 5 将转换为[0,0,0,0,0,1,0,0,0,0]"""
    data = [0,0,0,0,0,0,0,0,0,0]
    data[num] = 1
    return data


def todata():
    """ 从图片中加载数据并人工加标签"""
    # 检查路径是否存在
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # 读图片数据
    data = []
    items = os.listdir(imgpath)
    for item in items:
        filename = os.path.join(imgpath,item)
        print(filename)
        # img 是 28*28 的数组
        img = readimg.ImageToMatrix(filename)
        # 显示图片
        pltinit()
        plt.imshow(img, cmap='Greys')
        # plt.savefig('./img/t.png')
        plt.show()
        lable = input('请输入图片标签\n')
        lable = int(lable)
        if lable < 0:
            continue
        else:
            img = img.reshape(1,784)
            lable = np.resize(numtohot(lable), (1,10))
            it = np.append(img, lable)
            data = np.append(data, it)
    data.tofile(os.path.join(savepath,savename))

def showdata():
    # 加载数据集
    data = np.fromfile(os.path.join(savepath, "train.bin"),dtype=float)

    img1 = data[1][0:784]
    img1 = img1.reshape(28,28)
    pltinit()

    plt.imshow(img1, cmap='Greys')
    plt.show()


def loaddata(path):
    # 加载数据集
    data = np.fromfile(path,dtype=float)

    num = int(data.size/794)
    data = data.reshape(num,794)
    x=[]
    y=[]
    for i in range(num):
        img = data[i][0:784]
        img = img.reshape(28,28)
        lab = data[i][784:794]
        x = np.append(x,img)
        y = np.append(y,lab)
    x = x.reshape(num,784)
    y = y.reshape(num,10)

    return x,y

def checkdata(path,num):
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # batch = mnist.train.next_batch(50)
    imgs, labs = loaddata(path)
    for id in range(len(imgs)):
        pltinit()
        plt.imshow(imgs[id].reshape(28,28), cmap='Greys')
        plt.show()
        for i in range(10):
            if labs[id][i]>0:
                print(i)

        time.sleep(0.5)


imgpath = './myimg'
savepath = './mydata/'
savename = 'train2.bin'
if __name__ == '__main__':

    checkdata('./tmp/data/1.bin',12)
    # todata()

