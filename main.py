import cv2
import sys
from PyQt5 import uic
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from logger import log

id2chiclass = {1: '佩戴了口罩', 0: '未佩戴口罩'}
colors = ((255, 0, 0), (0, 255, 0))


def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印汉字
    fontsize = int(min(img.shape[:2])*0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    y = point[1]-font.getbbox(text)[1]
    if y <= font.getbbox(text)[1]:
        y = point[1]+font.getbbox(text)[1]
    draw.text((point[0], y), text, color, font=font)
    img = np.asarray(pilimg)
    return img


class Main_window(QLabel):
    def __init__(self):
        super().__init__()
        self.run_flag = False  # 运行标志位
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 开启摄像头   0代表使用电脑自带的摄像头，1代表其他的摄像头，如果有的话。
        self.ui = uic.loadUi('designer.ui')  # 动态加载ui界面
        self.model = torch.hub.load('./', 'custom', path='runs/train/exp5/weights/best.pt', source='local')
        self.ui.window.setPixmap(QPixmap("pictures/logo.png"))                # 待机页面
        self.ui.setWindowIcon(QIcon('pictures/logo.ico'))                     # ico图标
        self.ui.window.setScaledContents(True)  # 图片自适应大小
        self.ui.Timer = QtCore.QTimer()          # 定时器
        self.ui.Timer.timeout.connect(self.updateData)  # 信号与槽连接
        self.ui.start.clicked.connect(self.start_stop)  # 连接按键与事件
        self.ui.shut_down.clicked.connect(self.shut_down)
        self.ui.show()              # 启动显示

    def start_stop(self):        # 启动和停止按键
        self.run_flag = not self.run_flag
        if self.run_flag:
            self.ui.start.setText("停止检测")
            self.ui.Timer.start(50)
        else:
            self.ui.start.setText("开始检测")
            self.ui.Timer.stop()
            self.ui.window.setPixmap(QPixmap("pictures/logo.png"))

    def shut_down(self):      # 关闭系统按键
        log.info("system down.")
        self.ui.Timer.stop()  # 关闭定时器
        self.ui.close()

    def updateData(self):  # 更新主页面
        ret, show = self.cap.read()
        if ret:  # 采集到画面
            show = self.detection(show)
            show = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.window.setPixmap(QtGui.QPixmap.fromImage(show))

    def detection(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)
        # print(results.pandas().xyxy[0].to_numpy())# tensor-to-numpy
        results_ = results.pandas().xyxy[0].to_numpy()
        for box in results_:
            log.info(box)
            l, t, r, b = box[:4].astype('int')
            confidence = str(round(box[4] * 100, 2)) + "%"
            cls_name = box[6]
            # label_predict = box[5]
            cv2.rectangle(frame, (l, t), (r, b), colors[box[5]], 2)
            
            frame = puttext_chinese(frame, id2chiclass[box[5]], (l, t), colors[box[5]])  # 打印中文

            # cv2.putText(frame, cls_name + "-" + confidence, (l, t), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        return frame


if __name__ == "__main__":
    app = QApplication([])
    windows = Main_window()
    sys.exit(app.exec_())
