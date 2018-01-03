import tkinter as Tk
from tkinter import messagebox
import os
import tensorflow as tf
import numpy as np
import readimg
import common
import imgtodata
from PIL import Image, ImageTk



class Application(object):
    def __init__(self, master=Tk.Tk(), photonum=0):
        self.master = master
        self.master.geometry('263x400')
        self.photonum = photonum
        self.attrinit()
        # left
        self.frmL = Tk.Frame(master, bg='#f8fff8')
        Tk.Label(self.frmL, height=1, width=10, bg='#CCCCCC', text='原图').grid(row=0,column=0)
        self.canvas = Tk.Canvas(self.frmL,  height=150, width=150,bg='#ffffff')
        self.canvas.grid(row=1,column=0)
        self.canvas.create_image((75, 75), image=PhotoImages[self.picidx], tags='org')

        Tk.Label(self.frmL, height=1, width=10, bg='#CCCCCC', text='存储图').grid(row=2,column=0)
        self.canvas1 = Tk.Canvas(self.frmL,  height=150, width=150,bg='#ffffff')
        self.canvas1.grid(row=3,column=0)
        self.canvas1.create_image((75, 75), image=self.dealphoto, tags='deal')

        self.frmL.grid(row=0, column=0)

        # right
        self.frmR = Tk.Frame(master, bg='#f8fff8')
        self.frmR.lable = Tk.Label(self.frmR, text="共"+str(self.photonum)+"张,第"+str(self.picidx)+"张")
        self.frmR.lable.grid(row=0, column=0)

        self.frmR.lable1 = Tk.Label(self.frmR, text="已处理"+str(self.currentidx)+"张,跳过"+str(len(self.skiplist))+"张")
        self.frmR.lable1.grid(row=1, column=0)
        self.frmR.text = Tk.Text(self.frmR,height=2,width=15)
        self.frmR.text.grid(row=2,column=0)

        self.frmR.button_pre = Tk.Button(self.frmR, height=1, width=10, text="上一张", command=self.pre)
        self.frmR.button_pre.grid(row=3,column=0)
        self.frmR.button_next = Tk.Button(self.frmR, height=1, width=10, text="下一张", command=self.next)
        self.frmR.button_next.grid(row=4,column=0)
        self.frmR.button_skip = Tk.Button(self.frmR, height=1, width=10, text="跳过", command=self.skip)
        self.frmR.button_skip.grid(row=5,column=0)
        self.frmR.button_save = Tk.Button(self.frmR, height=1, width=10, text="保存", command=self.save)
        self.frmR.button_save.grid(row=6,column=0)
        self.frmR.grid(row=0, column=1, sticky='N')

        self.frmP = Tk.Frame(self.master)
        Tk.Label(self.frmP, width=30, height=1, text='预测模块').grid(row=0, column=0)
        self.frmP.btn1 = Tk.Button(self.frmP, height=1, width=10, text='批量预测', command=self.predictbatch)
        self.frmP.btn1.grid(row=1, column=0,  sticky='N')
        self.frmP.lable = Tk.Label(self.frmP, width=20, height=1, text='结果')
        self.frmP.lable.grid(row=2, column=0)
        self.frmP.scrb = Tk.Scrollbar(self.frmP)
        self.frmP.scrb.canvas = Tk.Canvas(self.frmP.scrb, bg='#ffffff')
        self.frmP.scrb.canvas.grid(row=0, column=0)
        self.frmP.scrb.grid(row=3, column=0)
        self.frmP.scrb.config(command=self.frmP.scrb.canvas.yview)
        self.frmP.scrb.canvas.config(yscrollcommand=self.frmP.scrb.set, height=220, width=400)
        self.frmP.grid(row=0, column=2, sticky='N')

    def attrinit(self):
        self.currentidx = 0
        self.picidx = 0
        self.skiplist = []
        self.data = None
        self.tf = {'sess': None, 't_conv': None, 'module_file': None,
                   'keep_prob': None, 'accuracy': None, 'x': None, 'y_': None,
                   'train_step': None, 'correct_prediction': None, 'module_file': None}
        self.predict_data={'images': None, 'labs': None}
        self.predict_result = {'result': None, 'wronglist': [], 'currentidx': 0}
        self.isload = None
        self.dealphoto = ImageTk.PhotoImage(
            Image.fromarray(255 - readimg.load_img_to_center1(imgs[self.picidx])).resize((150, 150), 5))
        self.savepath = './tmp/data'
        self.savename = 'train.bin'

    def pre(self):
        if self.currentidx <= 0:
            messagebox.showerror('消息框', '已经是第一张了')
            return

        self.currentidx = self.currentidx - 1
        self.picidx = self.picidx - 1
        while self.picidx in self.skiplist:
            self.picidx = self.picidx - 1
        self.frmR.lable1['text'] = "已处理" + str(self.currentidx) + "张,跳过" + str(len(self.skiplist)) + "张"

        self.data = np.delete(self.data,self.currentidx,0)
        self.canvas.delete('deal')
        self.canvas.delete('org')
        # self.canvas.create_rectangle(0, 0, 30, 30, fill='#000000',tags='s')
        self.canvas.create_image((75, 75), image=PhotoImages[self.picidx], tags='org')
        self.dealphoto = ImageTk.PhotoImage(
            Image.fromarray(255 - readimg.load_img_to_center1(imgs[self.picidx])).resize((150, 150), 5))
        self.canvas1.create_image((75, 75), image=self.dealphoto, tags='deal')
        self.frmR.lable['text'] = "共" + str(self.photonum) + "张,第" + str(self.picidx) + "张"

    def next(self):
        contents = self.frmR.text.get('1.0', Tk.END)
        try:
            t = int(contents)
            if t > 9 or t < 0:
                messagebox.showerror('消息框', '请输入0-9之间的数字')
                return
        except ValueError as e:
            messagebox.showerror('消息框', '请输入0-9之间的数字')
            return

        # 加载单条数据到data
        if self.currentidx < self.photonum:
            self.todata()

        self.currentidx = self.currentidx+1
        self.picidx = self.picidx + 1
        while self.picidx in self.skiplist:
            self.picidx = self.picidx+1
        # 保存到磁盘
        if self.currentidx == self.photonum:
            if self.currentidx == self.photonum:
                self.save()
        # 显示下一条数据
        if self.currentidx < self.photonum:
            self.canvas.delete('deal')
            self.canvas.delete('org')
            # self.canvas.create_rectangle(0, 0, 30, 30, fill='#000000',tags='s')
            self.canvas.create_image((75,75),image=PhotoImages[self.picidx],tags='org')
            self.dealphoto = ImageTk.PhotoImage(
                Image.fromarray(255 - readimg.load_img_to_center1(imgs[self.picidx])).resize((150, 150), 5))
            self.canvas1.create_image((75,75),image=self.dealphoto,tags='deal')
            self.frmR.lable['text'] = "共" + str(self.photonum) + "张,第" + str(self.picidx) + "张"
            self.frmR.lable1['text'] = "已处理" + str(self.currentidx) + "张,跳过" + str(len(self.skiplist)) + "张"
            # t = Image.fromarray(255 - readimg.load_img_to_center1(imgs[self.currentidx])).resize((150, 150), 5)
            # readimg.pltinit()
            # readimg.plt.imshow(np.array(t),'gray')
            # readimg.plt.show()

    def skip(self):
        """跳过"""
        r = messagebox.askokcancel('消息框', '是否跳过该图，不可撤销')
        if not r:
            return
        self.skiplist.append(self.picidx)
        self.picidx = self.picidx + 1

        # 显示
        self.canvas.delete('deal')
        self.canvas.delete('org')
        # self.canvas.create_rectangle(0, 0, 30, 30, fill='#000000',tags='s')
        self.canvas.create_image((75, 75), image=PhotoImages[self.picidx], tags='org')
        self.dealphoto = ImageTk.PhotoImage(
            Image.fromarray(255 - readimg.load_img_to_center1(imgs[self.picidx])).resize((150, 150), 5))
        self.canvas1.create_image((75, 75), image=self.dealphoto, tags='deal')
        self.frmR.lable['text'] = "共" + str(self.photonum) + "张,第" + str(self.picidx) + "张"
        self.frmR.lable1['text'] = "已处理" + str(self.currentidx) + "张,跳过" + str(len(self.skiplist)) + "张"

    def todata(self):
        """ 从图片中加载数据并人工加标签"""

        # img 是 28*28 的数组
        img = readimg.load_img_to_center1(imgs[self.picidx])

        lable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        lable[int(self.frmR.text.get('1.0',Tk.END))] = 1
        img = img.reshape(1, 784)
        lable = np.resize(lable, (1, 10))
        t = np.append(img,lable)
        if self.data is None:
            self.data = t
        else:
            self.data = np.row_stack((self.data,t))

    def save(self):
        r = messagebox.askokcancel('消息框', '是否覆盖'+os.path.join(self.savepath, self.savename)+'')
        if not r:
            return
        if self.data is None:
            messagebox.showerror('消息框','没有任何数据需要保存')
            return
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.data.tofile(os.path.join(self.savepath, self.savename))

    def predictbatch(self):
        # 评估
        if self.isload == None:
            self.tf['sess'], self.tf['y_conv'], self.tf['x'], self.tf['y_'], self.tf['keep_prob'], self.tf['train_step'], \
            self.tf['correct_prediction'], self.tf['accuracy'] = common.preduce()

            saver = tf.train.Saver()
            try:
                # arg:获取最近一次保存的变量文件名称
                self.tf['module_file'] = tf.train.latest_checkpoint('./tmp')
                print(self.tf['module_file'])
                saver.restore(self.tf['sess'], self.tf['module_file'])
                print("模型加载完毕")
                self.predict_data['images'], self.predict_data['labs'] = imgtodata.loaddata('./tmp/data/1.bin')
                self.predict_data['labs'] = self.predict_data['labs'].tolist()
            except:
                pass
            self.isload = True
        print('评估开始')

        self.predict_result['result'] = self.tf['y_conv'].eval(feed_dict={
            self.tf['x']: self.predict_data['images'], self.tf['keep_prob']: 1.0}).tolist()

        a = len(self.predict_result['result'])
        for i in range(a):
            predi = self.predict_result['result'][i].index(max(self.predict_result['result'][i]))
            lab = self.predict_data['labs'][i].index(max(self.predict_data['labs'][i]))
            if lab != predi:
                self.predict_result['wronglist'].append(i)
        b = len(self.predict_result['wronglist'])
        c = (a-b)/a*100
        t = "精度为：%0.3f" % c
        print(t+'%')
        self.frmP.lable['text'] = t+'%'

    def showwrongnext(self):
        if self.isload == None:
            messagebox.showerror('消息框', '请预测后在点击')
        img = self.predict_data['images'][self.predict_result['wrong']]

        self.wrongpix = ImageTk.PhotoImage(
                Image.fromarray(255 - self.predict_data['images'][self.wronglist[i]]).resize((150, 150), 5))
        self.frmP.scrb.canvas.create_image((75,75),image=PhotoImages[self.picidx],tags='org')
# 图片路径
path = './img/1'
items = os.listdir(path)
PhotoImages = []
imgs = []
for item in items:
    im = Image.open(os.path.join(path, item))
    img = ImageTk.PhotoImage(im)  # .resize((150, 150), 5)
    imgs.append(im)
    PhotoImages.append(img)
app = Application(photonum=len(imgs))
# 设置窗口标题:
app.master.title('标记')
# 保存路径
app.savepath = './tmp/data'
# 保存文件名
app.savename = '2.bin'
# 主消息循环:
app.master.mainloop()
