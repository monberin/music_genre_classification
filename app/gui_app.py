import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array

from amgc_app import get_f1

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Choose the mediafile'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.openFileNameDialog()

        self.prediction()

        
        self.show()
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","(*.png)", options=options)
    
    def prediction(self):
        model = keras.models.load_model('./finalized_model', custom_objects={'get_f1':get_f1})
        image_data = load_img(self.fileName,color_mode='rgba',target_size=(256,256))
        image = img_to_array(image_data)
        image = np.reshape(image, (1,256,256,4))
        pred = model.predict(image/255)
        pred = pred.reshape((7,))
        class_label = np.argmax(pred)
        print(class_label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())