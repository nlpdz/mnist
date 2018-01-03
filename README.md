###########CNN的手写字体识别##############
#1.	本程序的主体（模型的训练）是基于tensorflow中文社区的教程写的http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html

#2. 功能说明
	A.本程序按照教程实现了手写体模型的训练，和精度的测试，就给定的10000张测试图片，识别率为99.2%。
	B.实现了拍摄图片的单张预测（输入一张图片，输出预测的结果）
	C.实现了数据集的制作，即将一批图片做标记并存入二进制文件。
	D.实现了批量图片的精度检测，（将C.制作的数据集进行精度检测）

./tmp文件夹下的模型训练了33200次，

#3 文件说明
	image文件夹：
		mnist数据集中的部分数据图片
	img:
		供测试的一些图片
		1文件夹为手写的1000张图片，有网上找的图和手写拍照的图
	MNIST_data:
		mnist数据集
	tmp:
		模型文件 和 自制数据集存放位置

	changeIntoImg.py：
		将mnist中的数据转为图片，存储在当前目录下的  image  文件夹下，image文件夹下只有6000多张数据集中的图片，并未完全导出
	common.py：
		tensorflow 的一些变量的定义和模型的训练过程（可以忽略，全来自教程）
	GUI.py：
		数据集制作的图形界面（比较丑，勿喷，其他功能还未集成到GUI中）
		任意大小的图片将经过：灰度图转换->背景的初步剔除->识别主体移动到正中央->缩放到28*28像素大小，之后和标签一起存入到文件中
		数据保存前的格式为numpy对象,dtype=float,一行表示一条数据 1*794个float 前784为图像(28*28的二维变成一维数组) 后10个为标签 标签若为5 则为[0,0,0,0,0,1,0,0,0,0].
	imgtodata.py：（可忽略）
		GUI.py的前一个版本，控制台输入的，不过有个检验输入的标签是否正确的函数，所以还是保留了。
	MNIST.py：
		教程中的非CNN的识别demo
	mnist_cnn.py:
		模型的训练主体，同时包含精度的检测
		flags = 'train'表示训练
		flags = 'assessment' 表示精度的检测
	predict.py:
		预测某文件夹下的图片，修改path = './img/1'即可
		若控制台执行，会弹出图片的框，关闭后才会出预测结果，若IDE会sleep 1s后执行。
	predictbatch.py：
		批处理预测-检测自制数据集的精度用的。
	readimg.py：
		读取图片的工具
#4.	如何使用
	安装python3.5版本（建议直接官网下载对应的exe文件）
	安装所需的包
	控制台（CMD）下
	pip install tensorflow numpy matplotlib Pillow
	若下载的慢 可以换pip源
	pip install tensorflow numpy matplotlib Pillow -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

	控制台下直接输入python 文件 即可
	e.g.		python GUI.py
	程序均可直接执行，都有默认参数，路径可修改，见文件内部变量，若执行时报缺少包XX，pip install XX

#5 	如有问题 可联系我 1073280512@qq.com 猫先生和781137149<zhouzhouxxx@qq.com> wolf
