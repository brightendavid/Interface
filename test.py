#!/usr/bin/env python
# -*- coding:utf-8 -*-
from tkinter import *

import torch
from PIL import Image, ImageTk
import cv2
from tkinter import Button
from tkinter import filedialog
import tkinter as tk
import cpu_port
import one_pic_port

# 加载34s
# 不加载37s
# 大概率可用
# from two_stage_model import UNetStage1 as Net1
# from two_stage_model import UNetStage2 as Net2
# device = torch.device("cpu")
# model_path1 = r'C:\Users\brighten\Desktop\信息test/stage1.pth'
# model_path2 = r'C:\Users\brighten\Desktop\信息test/stage2.pth'
# checkpoint1 = torch.load(model_path1, map_location=device)
# checkpoint2 = torch.load(model_path2, map_location=device)
# model1 = Net1().to(device)
# model2 = Net2().to(device)
#
# model1.load_state_dict(checkpoint1['state_dict'])
# model2.load_state_dict(checkpoint2['state_dict'])
# model1.eval()
# model2.eval()



class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master, bg='#f0f0f0')
        self.flag = True
        self.pack(expand=YES, fill=BOTH)
        self.window_init()
        self.createWidgets()

    def window_init(self):
        self.master.title('图像拼接篡改检测工具')
        self.master.bg = 'GRAY'

        width, height = self.master.maxsize()

        self.master.minsize(width, height)
        self.master.geometry("{}x{}".format(width, height))
        # self.master.attributes("-alpha", 0.95)

    def createWidgets(self):
        # 首先定义存放结果的变量
        self.output1huidu = cv2.imread("test_result/stage1/1-1.bmp")
        self.output1 = cv2.imread("1-1st1.bmp")
        self.output2 = cv2.imread("1-1st2.bmp")
        self.output2_huidu = cv2.imread("test_result/stage2/1-1.bmp")

        # fm1  标题   在一层
        self.fm1 = Frame(self, bg='#f0f0f0')
        self.titleLabel = Label(self.fm1, text="图像拼接篡改检测工具", font=('微软雅黑', 48), fg="black",
                                bg='#f0f0f0')
        self.titleLabel.pack()
        self.fm1.pack(side=TOP, expand=YES, fill='x', pady=10)

        self.fm2 = Frame(self, bg='#f0f0f0')
        # fm2  的子对象
        self.fm2_left = Frame(self.fm2, bg='#f0f0f0')
        self.fm2_right = Frame(self.fm2, bg='#f0f0f0')
        self.fm2_left_top = Frame(self.fm2_left, bg='#f0f0f0')
        self.fm2_left_bottom = Frame(self.fm2_left, bg='#f0f0f0')
        self.fm2_right_top = Frame(self.fm2_right, bg="#f0f0f0")
        self.fm2_right_bottom = Frame(self.fm2_right, bg="#f0f0f0")

        # 单选按钮的   Frame23   在frame2   frame3  之间  图片总FRAME  和按钮总FRAME 之间
        self.fm23 = Frame(self, bg='#f0f0f0')
        self.fm23_right = Frame(self.fm23, bg='BLACK')
        self.fm23_left = Frame(self.fm23, bg='BLACK')
        # todo 单选按钮
        self.var1 = tk.BooleanVar()
        self.xianshishezhi = tk.Checkbutton(self.fm23_right, text="显示为假彩色图", variable=self.var1,
                                            command=self.show_the_caise, fg="black",
                                            bg='#f0f0f0', font=('微软雅黑', 20))
        self.xianshishezhi.pack(side=RIGHT)

        # todo  选择cpu和GPU按钮
        self.var2 = tk.BooleanVar()
        self.cpu_gpu = tk.Checkbutton(self.fm23_left, text="使用GPU", variable=self.var2, fg="black",
                                      bg='#f0f0f0', font=('微软雅黑', 20))
        self.cpu_gpu.pack(side=RIGHT)

        # 原始图片   fm3 的子对象  3个
        # fm3  图片展示框  给底色
        self.fm3 = Frame(self, bg="#f0f0f0")
        load1 = Image.open("test_data/1-1.png")
        load1 = load1.resize((400, 300))
        self.fm3_left = Frame(self.fm3, bg="#f0f0f0")
        global init_src
        init_src = ImageTk.PhotoImage(load1)

        self.src_label = Label(self.fm3_left, text="原图", image=init_src, bg="#f0f0f0", compound='top',
                               font=('微软雅黑', 20), fg='#f0f0f0')
        self.src_label.pack(side=LEFT, expand=YES, fill=BOTH, padx=50)
        self.fm3_left.pack(side=LEFT, expand=YES)

        load1 = Image.open("test_result/stage2/1-1.bmp")
        load1 = load1.resize((400, 300))
        self.fm3_left_right = Frame(self.fm3, bg="#f0f0f0")
        global init_stage2
        init_stage2 = ImageTk.PhotoImage(load1)
        self.stage2_label = Label(self.fm3_left_right, text="精确估计网络预测灰度图", image=init_stage2, bg="#f0f0f0",
                                  compound='top', font=('微软雅黑', 20),
                                  fg='#f0f0f0')
        self.stage2_label.pack(side=LEFT, expand=YES, fill=BOTH, padx=50)

        self.fm3_left_right.pack(side=RIGHT, expand=YES)


        self.output1huidu = cv2.imread("test_result/stage1/1-1.bmp")

        self.output1 = cv2.imread("1-1st1.bmp")

        # fm3结束
        # buttom1   显示原图
        self.srctButton = Button(self.fm2_left_top, text='选择并展示原图', bg='#f04a4c', fg='white',
                                 font=('微软雅黑', 15), width='25', command=self.show_src)
        self.srctButton.pack(side=RIGHT)

        # buttom2 显示阶段2 跳出弹窗
        self.truthButton = Button(self.fm2_right_top, text='展示精确估计网络结果', bg='#f04a4c', fg='white',
                                  font=('微软雅黑', 15), width='25', command=self.show_stage2)
        self.truthButton.pack(side=LEFT)
        self.truthButton.config(state=tk.DISABLED)

        # todo buttom 显示stage1
        self.show_stage2Buttom = Button(self.fm1, text="展示粗略估计网络结果", bg='#f04a4c', fg='white',
                                        font=('微软雅黑', 16), width='19', command=self.show_stage1)
        self.show_stage2Buttom.pack(side=RIGHT)

        self.fm2_left_top.pack(side=TOP, padx=60, pady=20, fill='x')

        self.fm2_left.pack(side=LEFT, padx=60, pady=20, expand=YES, fill='x')
        self.fm2_right.pack(side=RIGHT, padx=60, pady=20, expand=YES, fill='x')
        self.fm2_right_top.pack(side=TOP, padx=30, pady=20, fill='x')

        # 总体的父对象显示
        self.fm3.pack(side=TOP, padx=30, pady=20, fill='x')

        self.fm23_right.pack(side=LEFT, expand=YES)
        self.fm23_left.pack(side=RIGHT, expand=YES)
        self.fm23.pack(side=TOP, expand=YES, fill="x")
        self.fm2.pack(side=TOP, expand=YES, fill="x")

        # label 显示解释框  label的父对象是self
        self.label = tk.Label(self,
                              text="粗略估计网络和精确估计网络的预测结果为灰度图, 颜色越接近白色表示越可能是篡改边缘。 \n为了方便用户观察图像，我们还提供了预测结果的假彩色图，\n颜色越接近红色表示越有可能是篡改边缘",
                              font=('微软雅黑', 15), width='72', fg="black", bg="#f0f0f0")
        self.label.pack(side=BOTTOM)

    def show_stage2(self):
        torch.cuda.empty_cache()
        self.truthButton["text"] = "展示精确估计网络结果"
        # 判定 cpu 或gpu
        if self.var2.get() == False:
            # 函数调用 加载模型  和传输结果
            self.output1huidu, self.output1, self.output2, self.output2_huidu = cpu_port.one_picture_port(self.B)
            cv2.imwrite("stage1.png", self.output1)
            cv2.imwrite("stage2.png", self.output2)
            cv2.imwrite("stage1_gray.png", self.output1huidu)
            cv2.imwrite("stage2_gray.png", self.output2_huidu)
        else:
            self.output1huidu, self.output1, self.output2, self.output2_huidu = one_pic_port.one_picture_port(self.B)
        # 一阶段灰度   一阶段彩色  二阶段彩色  二阶段灰度
        # 消耗大量时间
        global init_stage2
        # 彩色图和灰度图公用一个全局变量
        # 实用get()方法  调用var1 的数值
        if self.var1.get() == True:
            image1 = Image.fromarray(cv2.cvtColor(self.output2, cv2.COLOR_BGR2RGB))  # 彩色显示
            load1 = image1
            load1 = load1.resize((400, 300))
            init_stage2 = ImageTk.PhotoImage(load1)
        else:
            image1 = Image.fromarray(cv2.cvtColor(self.output2_huidu, cv2.COLOR_BGR2RGB))  # 灰度显示
            load1 = image1
            load1 = load1.resize((400, 300))
            init_stage2 = ImageTk.PhotoImage(load1)
        self.stage2_label.configure(image=init_stage2)

    # todo 点击按钮  显示彩色或者  灰度
    def show_the_caise(self):
        # self.var1  返回 True  False boolean  表示是否显示假彩色图
        # print(self.var1.get())
        global init_stage2
        global load1
        if self.var1.get() == True:
            image1 = Image.fromarray(cv2.cvtColor(self.output2, cv2.COLOR_BGR2RGB))  # 彩色显示
            load1 = image1
            load1 = load1.resize((400, 300))
            init_stage2 = ImageTk.PhotoImage(load1)
        else:
            image1 = Image.fromarray(cv2.cvtColor(self.output2_huidu, cv2.COLOR_BGR2RGB))  # 灰度显示
            load1 = image1
            load1 = load1.resize((400, 300))
            init_stage2 = ImageTk.PhotoImage(load1)
        self.stage2_label.configure(image=init_stage2)
        self.update()

    def show_src(self):
        torch.cuda.empty_cache()
        self.truthButton.config(state=tk.ACTIVE)
        # todo 再次激活按钮  显示精确估计网络
        self.truthButton["text"] = "开始"
        # 按钮1 响应  展示  原图
        # 读取原图
        file_path = filedialog.askopenfilename()  # 调用系统文件读取函数
        self.A = Image.open(file_path)
        # fm6 展示原图
        # todo 加入进度条
        global init_src
        a, b = self.A.size
        self.B = self.A.copy()
        # 维护界面稳定性 过大的图片resize  到长宽都在400以内。很重要
        while a > 400 or b > 400:
            a = a // 2
            b = b // 2
            self.A = self.A.resize((a, b))
        # print(self.A.size)
        self.A = self.A.resize((400, 300))
        init_src = ImageTk.PhotoImage((self.A))
        self.src_label.configure(image=init_src)

        # # todo 在读取完成图片后  stage1 2 的显示框变化为空白  下面的代码，根据是否展示gt进行注释  若需要展示gt ，则注释掉；否则不注释，在读取图片之后，预测结果变为纯白图片
        self.output2_huidu = Image.open("white_block.PNG")  # Image读取
        self.output2_huidu = self.output2_huidu.resize((400, 300))
        self.output1 = cv2.imread("white_block.PNG")
        self.output1huidu = cv2.imread("white_block.PNG")
        global init_stage2
        load2 = self.output2_huidu
        init_stage2 = ImageTk.PhotoImage(load2)
        self.stage2_label.configure(image=init_stage2)

    def show_stage1(self):
        torch.cuda.empty_cache()
        if self.var1.get() == True:
            image2_stage1 = Image.fromarray(cv2.cvtColor(self.output1, cv2.COLOR_BGR2RGB))
        else:
            image2_stage1 = Image.fromarray(cv2.cvtColor(self.output1huidu, cv2.COLOR_BGR2RGB))
        image2_stage1 = image2_stage1.resize((400, 300))
        self.newwin = tk.Toplevel()
        self.newwin.iconbitmap(default=r'icon.ico')
        self.newwin.title("粗略估计网络展示")  # 设置窗口的标题
        self.newwin.geometry("900x400+600+300")  # 设置窗口的位置及大小

        self.newwin.fm = Frame(self.newwin, bg='#f0f0f0')

        global init_stage1  # stag1 的共有全局变量
        load2 = image2_stage1
        load2 = load2.resize((400, 300))
        init_stage1 = ImageTk.PhotoImage(load2)
        self.stage1_label = Label(self.newwin.fm, text="粗略估计网络假彩色预测图", image=init_stage1, bg='#f0f0f0', compound='top',
                                  font=('微软雅黑', 10), fg='#f0f0f0')
        self.stage1_label.pack()

        self.newwin.fm.pack(side=LEFT, expand=YES, fill=BOTH, pady=10)


if __name__ == '__main__':
    app = Application()
    app.mainloop()
