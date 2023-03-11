import numpy as np
import cv2, sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from save_img_ui import Ui_Form
import time
import serial.tools.list_ports


class App(QWidget, Ui_Form):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.thread = VideoThread()
        self.thread.start()
        self.thread.cam.connect(self.update_image_cam)
        # self.thread.measure_value.connect(self.save_img)
        self.thread_2 = tool_Thread()
        self.thread_2.start()
        self.thread_2.measure_value.connect(self.save_img)
        # self.thread_2.cam.connect(self.update_image_cam)
        self.ui.toolButton.clicked.connect(self._open_file_dialog)
        self.dir = str()
        # self.ui.pushButton_save.clicked.connect(self.save_img)
        self.cv_img = None
        self.result = time.localtime()
        self.save_img_name = str(self.result.tm_year) + "_" + str(self.result.tm_mon) \
                             + "_" + str(self.result.tm_mday) + "_" + str(self.result.tm_hour) + "_" + str(self.result.tm_min)
        self.i = 0

    def save_img(self, bo):
        name = self.save_img_name + '_' + str(self.i) + '.png'
        dir = self.dir + '\\' + name
        status = cv2.imwrite(dir, self.cv_img)
        self.i += 1
        self.change_lable_save_img(name)

    def change_lable_save_img(self, name):
        text = "save_img_name:" + str(name)
        self.ui.img_save.setText(text)

    def _open_file_dialog(self):
        result = str(QFileDialog.getExistingDirectory())
        print(result)
        self.dir = result
        self.ui.img_save_2.setText(result)

    def update_image_cam(self, cv_img):
        self.cv_img = cv_img
        cv_img = self.resize_img(cv_img)
        w, h, l = cv_img.shape  # 圖像參數（高度、寬度、通道數）
        qt_img = self.convert_cv_qt(cv_img, w, h)
        self.ui.cam.setPixmap(qt_img)  # 顯示於Label中

    def resize_img(self, img):
        (w, h, l) = img.shape
        if int(w) > 180:
            scale = 2.0
            dim = (int(h / scale), int(w / scale))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def convert_cv_qt(self, cv_img, im_w, im_h):  # 輸入label
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 轉成RGB
        except:
            rgb_image = np.zeros((1280, 720, 3), np.uint8)
        h, w, ch = rgb_image.shape  # 取得參數，（高度、寬度、通道數）
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)  # 讀取圖片顯示在QLabel上
        p = convert_to_Qt_format.scaled(im_w, im_h, Qt.KeepAspectRatio)  # 固定長寬比
        return QPixmap.fromImage(p)  # 格式轉換Pixmap>Image

class tool_Thread(QThread):
    measure_value = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        while self._run_flag:
            self.serial_test()
            self.measure_value.emit('')

    def serial_test(self):
        COM_PORT = 'COM%s' % '3'
        BAUD_RATES = 57600
        BYTE_SIZE = 8
        PARITY = 'N'
        STOP_BITS = 1
        ser = serial.Serial(COM_PORT, BAUD_RATES, BYTE_SIZE, PARITY, STOP_BITS, timeout=None)
        string_slice_start = 8
        string_slice_period = 12
        try:
            while True:
                while ser.in_waiting:
                    data_raw = ser.read_until(b'\r')
                    data = data_raw.decode()
                    equipment_ID = data[:string_slice_start - 1]
                    altered_string = data[string_slice_start:string_slice_start + string_slice_period - 1]
                    altered_int = float(altered_string)
                    unit = list(data)
                    I = 'I'
                    if unit[(-2)] == I:
                        altered_unit = 'in'
                    else:
                        altered_unit = 'mm'
                    a = []
                    a.append(altered_int)
                    a.append(equipment_ID)
                    a.append(altered_unit)
                    print(a)
                    ser.close()
                    # return a
        except:
            pass


class VideoThread(QThread):
    cam = pyqtSignal(np.ndarray)
    measure_value = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture()
        # The device number might be 0 or 1 depending on the device and the webcam
        cap.open(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 70.0)  # 亮度 130
        cap.set(cv2.CAP_PROP_CONTRAST, 128.0)  # 對比度 32
        cap.set(cv2.CAP_PROP_SATURATION, 128.0)  # 飽和度 64
        cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色調 0
        cap.set(cv2.CAP_PROP_EXPOSURE, -5.5)  # 曝光 -4
        while self._run_flag:
            ret, cv_img = cap.read()
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
            self.cam.emit(cv_img)
            # self.serial_test()
            # self.measure_value.emit('')

    def serial_test(self):
        COM_PORT = 'COM%s' % '3'
        BAUD_RATES = 57600
        BYTE_SIZE = 8
        PARITY = 'N'
        STOP_BITS = 1
        ser = serial.Serial(COM_PORT, BAUD_RATES, BYTE_SIZE, PARITY, STOP_BITS, timeout=None)
        string_slice_start = 8
        string_slice_period = 12
        try:
            while True:
                while ser.in_waiting:
                    data_raw = ser.read_until(b'\r')
                    data = data_raw.decode()
                    equipment_ID = data[:string_slice_start - 1]
                    altered_string = data[string_slice_start:string_slice_start + string_slice_period - 1]
                    altered_int = float(altered_string)
                    unit = list(data)
                    I = 'I'
                    if unit[(-2)] == I:
                        altered_unit = 'in'
                    else:
                        altered_unit = 'mm'
                    a = []
                    a.append(altered_int)
                    a.append(equipment_ID)
                    a.append(altered_unit)
                    print(a)
                    ser.close()
                    # return a
        except:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
