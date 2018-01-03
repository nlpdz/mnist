from PIL import Image,ImageEnhance
import numpy as np

import cv2
from pylab import *
import os


def pltinit():
    fig = plt.gcf()
    fig.set_size_inches(0.28, 0.28)
    plt.xticks([])
    plt.yticks([])
    # 关闭坐标轴
    plt.axis('off')


def ImageToMatrix(filename):
    # 读取图片
    image = Image.open(filename)

    image = image.convert('L')

    image = contrast(image)

    image = bright(image,4).resize((28, 28), Image.ANTIALIAS)
    # image.show()
    image = contrast(image)
    # image.show()
    image = bright(image,1.5)
    # image.show()
    # image = image.convert('L')
    new_data = 255 - np.reshape(image.getdata(),(28, 28))

    # for i in range(28):
    #     for j in range(28):
    #         if new_data[i][j] < 100:
    #             new_data[i][j] = 0
    # new_im = Image.fromarray(new_data)
    # # # 显示图片
    # new_im.show()
    return new_data


def binary(image,threshold):
    """二值化图像"""
    image = image.convert('L')
    array = np.array(image)
    w = len(array[0])
    h = len(array)
    for i in range(h):
        for j in range(w):
            if array[i][j]<threshold:
                array[i][j] = 0
            else:
                array[i][j] = 255
    return array


def contrast(image):
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 6
    image_contrasted = enh_con.enhance(contrast)

    return image_contrasted


def bright(image,attr):
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = attr
    image_brightened = enh_bri.enhance(brightness)
    # image_brightened.show()
    return image_brightened


def batch_contrast(path):
    # 对比度增强
    for i in range(10):
        name = os.path.join(path, 'hand%d.jpg' % i)
        image = Image.open(name)
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image_contrasted = enh_con.enhance(contrast)
        name = os.path.join(path, 'hand%d.png' % i)
        image_contrasted.save(name)
        # image_contrasted.show()


def block(nparray):
    """输入二值化后的数组，找到最小的框"""
    top,bottom,left,right = 0,0,0,0

    for i in range(len(nparray)):
        if nparray[i].all():
            pass
        else:
            top = i - 1
            break
    idx = list(range(len(nparray)))
    idx.reverse()

    for i in idx:
        if nparray[i].all():
            pass
        else:
            bottom = i + 1
            break
    idx = list(range(len(nparray[0])))
    for i in idx:
        if nparray[:, i].all():
            pass
        else:
            left = i - 1
            break
    idx = list(range(len(nparray[0])))
    idx.reverse()
    for i in idx:
        if nparray[:, i].all():
            pass
        else:
            right = i + 1
            break
    if top == -1:
        top = 0
    if left == -1:
        left = 0
    return top,bottom,left,right


def stitch(imgarray,top,bot,left,right):
    """将top bot left right 确定的图片放到正中间"""
    rate = 0.7
    height = int((bot-top)/rate)
    width = int((right-left)/rate)
    if width > height:
        height = width
    margintop = int((height - (bot - top -1))/2)
    marginleft = int((height - (right - left -1))/2)
    w = right - left -1
    data = 255- np.zeros([height,height])
    for i in range(bot-top-1):
        data[i+margintop,marginleft:marginleft+w] = imgarray[top+i+1,left:right-1]
    data = 255 - data
    return data


def load_img_to_center(path):
    """加载原图，将原图二值化后移到正中间，返回一个正方形的图片数组"""
    img = cv2.imread(path, 0)  # 直接读为灰度图像
    image = bright(Image.fromarray(img), 2)
    ret2,th2 = cv2.threshold(np.array(image),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = bright(Image.fromarray(th2), 4)
    # pltinit()
    # plt.imshow(np.array(image),'gray')
    # plt.show()
    top,bottom,left,right = block(th2)

    # pltinit()
    # plt.imshow(img,'gray')
    # plt.show()

    img = stitch(np.array(image),top,bottom,left,right)
    img = Image.fromarray(img).resize((28, 28), 5)
    # img.show()
    return np.array(img)


def load_img_to_center1(image):
    """加载原图，将原图二值化后移到正中间，返回一个正方形的图片数组"""
    img = image.convert('L')
    image = bright(img, 1)
    ret2,th2 = cv2.threshold(np.array(image),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = bright(Image.fromarray(th2), 2)
    # pltinit()
    # plt.imshow(np.array(image),'gray')
    # plt.show()
    top,bottom,left,right = block(th2)

    # pltinit()
    # plt.imshow(img,'gray')
    # plt.show()

    img = stitch(np.array(image),top,bottom,left,right)
    img = Image.fromarray(img).resize((28, 28), 5)
    # img.show()
    return np.array(img)


if __name__ == "__main__":
    path = './img/2/2_02x01.bmp'

    pltinit()
    img = load_img_to_center(path)

    plt.imshow(img,'gray')
    plt.show()