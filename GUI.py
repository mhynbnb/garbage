'''
File : GUI.py
Auther : MHY
Created : 2024/6/24 12:48
Last Updated : 
Description : 
Version : 
'''
import sys

from PIL import Image,ImageQt
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QMessageBox
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QThread, pyqtSignal
# from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
import ui
import qdarkstyle
import threading
import cv2
import os
from ultralytics import YOLO
import time
disconncted = 0
disconncted2=0
detect_type=0
'''0: camara    1: image    2: dir  3: video'''
source='0'
save_flag=0
model = YOLO('garbage.pt')
model.predict('./assets/start.jpeg')
os.makedirs('results',exist_ok=True)
os.makedirs('results/video',exist_ok=True)
os.makedirs('results/pic',exist_ok=True)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stop_signal = pyqtSignal()
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480)
                self.change_pixmap_signal.emit(p)
            if disconncted!=0:
                break
        cap.release()
class detectThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stop_signal = pyqtSignal()
    cost_time_thread_signal = pyqtSignal(str)
    object_num_thread_signal = pyqtSignal(str)

    def run(self):
        time_start = time.time()
        if detect_type==0:
            cap = cv2.VideoCapture(0)
            if save_flag:
                video_name = './results/video/'+'camera_'+str(int(time.time()))+'.mp4'
                fps=30
                frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_name, fourcc, fps, frameSize)
            while True:
                ret, frame = cap.read()
                if ret:

                    result = model(frame)
                    result[0].save(filename="./assets/result.jpg")
                    self.object_num_thread_signal.emit(str(len(result[0].boxes)))
                    if save_flag:
                        out.write(cv2.imread('./assets/result.jpg'))

                    convert_to_qt_format = QImage('./assets/result.jpg')
                    self.change_pixmap_signal.emit(convert_to_qt_format)
                if disconncted2!=0:
                    break
            cap.release()
            if save_flag:
                out.release()
        elif detect_type==1:
            result = model(source)
            result[0].save(filename="./assets/result.jpg")
            self.object_num_thread_signal.emit(str(len(result[0].boxes)))
            if save_flag:

                cv2.imwrite('./results/pic/'+'image_'+str(int(time.time()))+'.jpg',cv2.imread('./assets/result.jpg'))
            convert_to_qt_format = QImage('./assets/result.jpg')
            self.change_pixmap_signal.emit(convert_to_qt_format)

        elif detect_type==2:
            if save_flag:
                base_forder=os.path.join('./results/pic/',os.path.basename(source))
                os.makedirs(base_forder,exist_ok=True)
            images = [f for f in os.listdir(source) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for image in images:
                result = model(os.path.join(source, image))
                result[0].save(filename="./assets/result.jpg")
                self.object_num_thread_signal.emit(str(len(result[0].boxes)))
                if save_flag:
                    cv2.imwrite(os.path.join(base_forder,image),cv2.imread('./assets/result.jpg'))
                convert_to_qt_format = QImage('./assets/result.jpg')
                self.change_pixmap_signal.emit(convert_to_qt_format)
                if disconncted2!=0:
                    break
        elif detect_type==3:
            # print(source)
            cap = cv2.VideoCapture(source)
            if save_flag:
                video_name = './results/video/'+'video_'+str(int(time.time()))+'.mp4'
                fps=30
                frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_name, fourcc, fps, frameSize)
            while True:
                ret, frame = cap.read()
                if ret:
                    result = model(frame)
                    result[0].save(filename="./assets/result.jpg")
                    self.object_num_thread_signal.emit(str(len(result[0].boxes)))
                    if save_flag:
                        out.write(cv2.imread('./assets/result.jpg'))
                    convert_to_qt_format = QImage('./assets/result.jpg')
                    self.change_pixmap_signal.emit(convert_to_qt_format)
                if disconncted2 != 0:
                    break
            cap.release()
            if not cap.isOpened():
                print("VideoCapture对象可能已释放或未成功打开。")
            if save_flag:
                out.release()
        self.cost_time_thread_signal.emit(f'{(time.time()-time_start):.5f} s')
class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 调用setupUi方法来初始化UI
        start_pix=QPixmap('./assets/start.jpeg').scaled(640, 480)
        self.label.setPixmap(start_pix)
        self.label_8.setText('Camera 0')
        self.camara_show_thread = VideoThread()
        self.detect_thread = detectThread()
        self.detect_thread.cost_time_thread_signal.connect(self.show_cost_time)
        self.detect_thread.object_num_thread_signal.connect(self.show_object_num)
        self.pushButton_6.clicked.connect(self.open_camara)
        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.open_dir)
        self.pushButton_3.clicked.connect(self.open_video)
        self.pushButton_4.clicked.connect(self.start_detect)
        self.pushButton_5.clicked.connect(self.stop_detect)
    def start_detect(self):
        #开始检测前先关闭摄像头，防止占用
        global disconncted2, detect_type, source, disconncted,save_flag
        save_flag=self.radioButton.isChecked()
        if self.camara_show_thread.isRunning():
            if disconncted==0:
                self.camara_show_thread.change_pixmap_signal.disconnect()
                disconncted=1
        if self.detect_thread.isRunning():
            if disconncted2==0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2=1

        # if detect_type==0 or detect_type==3:
        disconncted2=0
        self.detect_thread.change_pixmap_signal.connect(self.update_image)

        self.detect_thread.start()

    def open_camara(self):
        global disconncted, detect_type, source,disconncted2
        if self.camara_show_thread.isRunning():
            if disconncted == 0:
                self.camara_show_thread.change_pixmap_signal.disconnect()
                disconncted = 1
        if self.detect_thread.isRunning():
            if disconncted2 == 0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2 = 1

        detect_type=0
        self.label_8.setText('Camera 0')
        source='0'

        disconncted = 0
        self.camara_show_thread.change_pixmap_signal.connect(self.update_image)
        self.camara_show_thread.start()
    def update_image(self, cv_img):
        self.label.setPixmap(QPixmap.fromImage(cv_img).scaled(640, 480))
    def show_cost_time(self,time_cost):
        self.label_5.setText(time_cost)
    def show_object_num(self,num):
        self.label_6.setText(num)
    def open_image(self):
        global disconncted, detect_type, source,disconncted2
        detect_type=1
        if self.camara_show_thread.isRunning():
            if disconncted == 0:
                self.camara_show_thread.change_pixmap_signal.disconnect()
                disconncted = 1
        if self.detect_thread.isRunning():
            if disconncted2 == 0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2 = 1


        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
                                                   options=options)
        if file_name:  # 如果用户选择了文件
            pixmap = QPixmap(file_name)
            self.label.setPixmap(pixmap.scaled(640, 480))
            self.label_8.setText(file_name)
            source=file_name
        else:
            self.label_8.setText('no source')
            detect_type = 4
    def open_dir(self):
        global disconncted, detect_type, source, disconncted2
        detect_type=2
        if self.camara_show_thread.isRunning():
            if disconncted == 0:
                self.camara_show_thread.change_pixmap_signal.disconnect()
                disconncted = 1
        if self.detect_thread.isRunning():
            if disconncted2 == 0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2 = 1

        options = QFileDialog.Options()
        dir_name = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)
        if dir_name:  # 如果用户选择了文件
            images = [f for f in os.listdir(dir_name) if f.endswith(('.png', '.jpg', '.jpeg'))]
            selected_images = images[:4] if len(images) >= 4 else images
            images_to_show = self.create_image_grid(selected_images, dir_name)

            if images_to_show is not None:
                images_to_show.save('./assets/temp.jpg')
                pixmap = QPixmap('./assets/temp.jpg')
                self.label.setPixmap(pixmap)

            self.label_8.setText(dir_name)
            source=dir_name
        else:
            self.label_8.setText('no source')
            detect_type = 4

    def create_image_grid(self, images, folder_path):
        try:
            img_size = (320, 240)  # 假设每个小图的尺寸
            cols = 2
            rows = 2 if len(images) >= 4 else len(images) // cols + (len(images) % cols > 0)
            # 初始化空白画布
            grid_width = cols * img_size[0]
            grid_height = rows * img_size[1]
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            for i, img_name in enumerate(images):
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path).resize(img_size)
                x = (i % cols) * img_size[0]
                y = (i // cols) * img_size[1]
                grid_image.paste(img, (x, y))

            return grid_image
        except Exception as e:
            print(f"Error creating image grid: {e}")
            return None
    def open_video(self):
        global disconncted, detect_type, source, disconncted2
        detect_type=3
        if self.camara_show_thread.isRunning():
            if disconncted == 0:
                self.camara_show_thread.change_pixmap_signal.disconnect()
                disconncted = 1
        if self.detect_thread.isRunning():
            if disconncted2 == 0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2 = 1

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                                   "Video Files (*.mp4 *.avi *.mkv *.wmv);;All Files (*)",
                                                   options=options)
        if file_name:  # 如果用户选择了文件
            cap = cv2.VideoCapture(file_name)
            success, frame = cap.read()  # 读取第一帧
            cap.release()  # 释放资源

            if success:
                # 将OpenCV图像转换为QImage
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                # 转换为QPixmap并设置到标签
                pixmap = QPixmap.fromImage(q_image.rgbSwapped())  # 需要rgbSwapped()处理颜色通道
                self.label.setPixmap(pixmap)

            # self.label.setPixmap(pixmap.scaled(640, 480))
            self.label_8.setText(file_name)
            source=file_name
        else:
            self.label_8.setText('no source')
            detect_type = 4

    def stop_detect(self):

        global disconncted, detect_type, source, disconncted2
        if self.detect_thread.isRunning():
            if disconncted2 == 0:
                self.detect_thread.change_pixmap_signal.disconnect()
                disconncted2 = 1



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
