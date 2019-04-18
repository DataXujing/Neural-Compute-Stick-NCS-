'''
计算棒，手写数字识别 测试demo
xujing
2019-04-17
'''
# import tensorflow as tf
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PIL import ImageGrab, Image
# import cv2
import numpy as np

from mvnc import mvncapi as mvnc
import sys
import os
import datetime
import uuid

# glob variable
GRAPH_FILE = "./model/AlexNet.graph"
IMG_SIZE = 28

class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.resize(284*2, 330*2)  # resize设置宽高
        self.move(100, 100)    # move设置位置 说明在哪个位置截图
        self.setWindowIcon(QIcon('./logo.ico'))
        self.setWindowTitle('计算棒测试-手写数字识别')
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点

        # 添加一系列控件
        self.label_draw = QLabel('', self)

        self.label_draw.setStyleSheet("QLabel{background:rgb(255,255,255)}")
        self.label_draw.setGeometry(2, 2, 550, 550) # (x,y,width,height)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('预测：', self)
        self.label_result_name.setGeometry(2, 570, 61, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 570, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 570, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 570, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 570, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

        # 时间
        self.label_time_name = QLabel('识别时间：', self)
        self.label_time_name.setGeometry(320, 570, 100, 35)
        self.label_time_name.setAlignment(Qt.AlignCenter)

        self.label_time = QLabel(' ', self)
        self.label_time.setGeometry(430, 570, 110, 35)
        self.label_time.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_time.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_time.setAlignment(Qt.AlignCenter)

        # 计算棒信息
        self.label_ncs_name = QLabel('NCS状态：', self)
        self.label_ncs_name.setGeometry(2, 610, 100, 35)
        self.label_ncs_name.setAlignment(Qt.AlignCenter)

        self.label_ncs = QLabel(' ', self)
        self.label_ncs.setGeometry(110, 610, 430, 35)
        self.label_ncs.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_ncs.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_ncs.setAlignment(Qt.AlignCenter)

        # 打开计算棒的设备
        mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL,2)
        #获取连接到主机系统的神经计算设备列表
        self.devices = mvnc.enumerate_devices()

        if len(self.devices) == 0:
            print("[INFO] 未发现计算棒的任何设备！")
            raise("[Error] No devices found!")
            # quit()

        #调用第一个NCS设备
        self.device = mvnc.Device(self.devices[0])
        print("[INFO] 打开的计算棒设备id：" + str(self.devices[0]))
        #打开通信
        self.device.open() 
        self.ncs_info = "NCS调用成功，device ID:" + str(self.devices[0]) 

        # 加载图
        with open(GRAPH_FILE,mode='rb') as f:
            self.graphFileBuff = f.read()

        #初始化一个名为graph_name的图
        self.graph = mvnc.Graph("alexnet") 

        #创建输入和输出先进先出队列，将图加载到设备
        self.input_fifo,self.output_fifo=self.graph.allocate_with_fifos(self.device,self.graphFileBuff) 

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 650, 650)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
        im = im.convert('L')
        im.save('./save_pic/'+ str(uuid.uuid1()) + ".png")

        recognize_result,time_result= self.recognize_img(im)  # 调用识别函数
        
        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.label_time.setText(str(time_result))
        self.label_ncs.setText(str(self.ncs_info))
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.label_time.setText('')
        self.label_ncs.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()
        # 关闭队列
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        #关闭图和设备
        self.graph.destroy()
        self.device.close()
        self.device.destroy()


    def recognize_img(self, img):  # 手写体识别函数

        img_tensor_0 = img.convert('L')
        img_tensor_0 = 1 -  np.array(img_tensor_0,dtype=np.float32) / 255
        img_tensor_0 = img_tensor_0.reshape((1,28*28))

        temp = [0.0 if i - 0.05882353 <= 0.0000001 else i for i in list(img_tensor_0[0]) ]
        start_time = datetime.datetime.now()
        img_tensor = np.array(temp,dtype=np.float32) 

        # print(img_tensor)

        #将输入张量写入输入Fifo并将其排队以进行推理
        self.graph.queue_inference_with_fifo_elem(self.input_fifo,self.output_fifo,img_tensor,"user obj") 
        #推理完成后，使用Fifo.read_elem（）获取推理结果。
        output,user_obj = self.output_fifo.read_elem() 

        end_time = datetime.datetime.now()
         
        # print("[INFO] 识别结果：" + str(output)) 
        # print("[INFO] 运行时间：" + str(end_time-start_time))
        # print("[INFO] User Object: " + user_obj)
        print("[INFO] 识别结果：" + str(output))
        print("[INFO] 识别时间：" + str((end_time-start_time).total_seconds()))
     
        return np.argmax(output),(end_time-start_time).total_seconds()




