import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import matplotlib
from PyQt5.QtGui import QIcon
import sip

import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array

from amgc_app import get_f1

from pydub import AudioSegment
import matplotlib.pyplot as plt
import sklearn

import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

        self.generate_spectogram()

        self.prediction()

        
        self.show()
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #, "(*.mp3)", "(*.fvl)"]
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "select the media file","Audio Files(*.wav *.mp3 *.flv)", options=options)
        print(self.fileName)

    def generate_spectogram(self):

        import os
        filename, file_extension = os.path.splitext(self.fileName)

        # convert file to wav format
        if file_extension==".mp3":
            sound = AudioSegment.from_mp3(self.fileName)
            sound.export(filename+".wav", format="wav")
            self.fileName = filename+".wav"
        
        if file_extension==".flv":
            sound = AudioSegment.from_flv(self.fileName)
            sound.export(filename+".wav", format="wav")
            self.fileName = filename+".wav"


        # other way to do this:

        #sample_rate, samples = wavfile.read(self.fileName)
        #frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # plt.pcolormesh(times, frequencies, spectrogram)
        # plt.imshow(spectrogram)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        y,sr = librosa.load(self.fileName)
        mels = librosa.feature.melspectrogram(y=y,sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,ref=np.max))

        self.spect_path = filename+"-spectrogram"
        plt.savefig(self.spect_path)

    
    def prediction(self):
        model = keras.models.load_model('../finalized_model', custom_objects={'get_f1':get_f1})
        image_data = load_img(self.spect_path,color_mode='rgba',target_size=(256,256))
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