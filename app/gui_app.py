import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QVBoxLayout, QPushButton, QMessageBox,QLabel
import matplotlib
from PyQt5.QtGui import QIcon, QPalette, QColor, QPixmap
import sip

import numpy as np
import pandas as pd

from pydub import AudioSegment
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

import librosa
from librosa import feature, onset, beat, effects
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import joblib
import os

class SearchFiles(QWidget):

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
        self.generate_features()

        
    
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #, "(*.mp3)", "(*.fvl)"]
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "select the media file","Audio Files(*.wav *.mp3 *.flv)", options=options)
        print(self.fileName)

    def generate_features(self):

        fn_list_i = [
            feature.chroma_stft,
            feature.chroma_cens,
            feature.spectral_centroid,
            feature.spectral_bandwidth,
            feature.spectral_rolloff,
            feature.poly_features
            ]

        fn_list_ii = [
            feature.rms,
            feature.zero_crossing_rate,
            feature.spectral_flatness
            ]
            
        def get_feature_vector(y,sr): 

            header =[
            "chroma_cens",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "poly_features",
            "rmse",
            "zero_crossing_rate",
            "spectral_flatness",
            "spectral_centroids[0]",
            "spectral_centroids_delta",
            "spectral_centroids_accelerate",
            "spectral_bandwidth_2",
            "spectral_bandwidth_3",
            "spectral_bandwidth_4",
            "spectral_flux",
            "tempo_bpm",
            "harmonics",
            "perceptual_shock_wave"
            ]

            data = pd.read_csv('./features_new.csv')
            data = data.iloc[0:, 1:] 
            data.head()
            data = data.loc[:, data.columns != 'genre']

            feat_vect_i = [ np.mean(funct(y,sr)) for funct in fn_list_i]
            feat_vect_ii = [ np.mean(funct(y)) for funct in fn_list_ii] 
            feature_vector = feat_vect_i + feat_vect_ii 
            spectral_centroids = feature.spectral_centroid(y, sr=sr)[0]
            spectral_centroids_delta = np.mean(feature.delta(spectral_centroids))
            spectral_centroids_accelerate = np.mean(feature.delta(spectral_centroids, order=2))
            feature_vector.append(np.mean(spectral_centroids))
            feature_vector.append(spectral_centroids_delta)
            feature_vector.append(spectral_centroids_accelerate)

            # Spectral Bandwidth
            spectral_bandwidth_2 = np.mean(librosa.feature.spectral_bandwidth(y, sr=sr)[0])
            spectral_bandwidth_3 = np.mean(librosa.feature.spectral_bandwidth(y, sr=sr, p=3)[0])
            spectral_bandwidth_4 = np.mean(librosa.feature.spectral_bandwidth(y, sr=sr, p=4)[0])
            feature_vector.append(spectral_bandwidth_2)
            feature_vector.append(spectral_bandwidth_3)
            feature_vector.append(spectral_bandwidth_4)

            # spectral flux
            onset_env = np.mean(onset.onset_strength(y=y, sr=sr))
            feature_vector.append(onset_env)
            
            tempo_y, _ = beat.beat_track(y, sr=sr)
            feature_vector.append(np.mean(tempo_y))

            # Perceptrual shock wave
            y_harm, y_perc = effects.hpss(y)

            feature_vector.append(np.mean(y_harm))
            feature_vector.append(np.mean(y_perc))

            feature_vector = pd.DataFrame([feature_vector[1:]],columns=header)

            data = data.append(feature_vector)

            cols = data.columns
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(data)

            data = pd.DataFrame(np_scaled, columns = cols)

            return data.iloc[-1:]
        
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
            
        y , sr = librosa.load(self.fileName,sr=None)
        feature_vector = get_feature_vector(y, sr)

        loaded_rf = joblib.load("./svm.joblib")
        y_pred = loaded_rf.predict(feature_vector)
        show_result(y_pred[0])


def show_result(label):
    genres = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae','rock']
    messagebox = QMessageBox()
    messagebox.setWindowTitle('Result!')
    print(f"./images/{genres[label]}.png")
    messagebox.setIconPixmap(QPixmap(f"./images/{genres[label]}.png"))
    messagebox.setText('Predicted genre: '+genres[label])
    messagebox.exec_()


def search_files_window():
    ex = SearchFiles()


def help_info():
    messagebox = QMessageBox()
    messagebox.setWindowTitle("Info!")
    messagebox.setText('This app classifies a media file into 9 different genres: blues, classical music, country, disco, pop, hiphop, metal, reggae and rock.')
    messagebox.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    qp = QPalette()

    btextcolor = QColor(80, 80, 63)
    windowcolor = QColor(201, 201, 186)
    bcolor = QColor(169, 169, 137)

    qp.setColor(QPalette.ButtonText, btextcolor)
    qp.setColor(QPalette.Window,windowcolor)
    qp.setColor(QPalette.Button, bcolor)

    qp.setColor(QPalette.WindowText, btextcolor)
    qp.setColor(QPalette.Base,windowcolor)
    qp.setColor(QPalette.PlaceholderText, btextcolor)
    qp.setColor(QPalette.Text, btextcolor)
    qp.setColor(QPalette.BrightText,btextcolor)

    app.setPalette(qp)
    app.setStyle('Fusion')


    window = QWidget()
    window.resize(640,480)
    window.setWindowTitle("genre prediction :)")
    layout = QVBoxLayout()


    button_style = "QPushButton {\
                        background-color: #989871;\
                        color: #50503F;\
                        font: bold 16px;\
                        height: 10em;\
                        position: center;\
                    }\
                    QPushButton:hover {\
                        background-color: #A9A989;\
                    }"

    button = QPushButton('choose a mediafile!')
    button.setStyleSheet(button_style)

    info_button = QPushButton('info!')
    info_button.setStyleSheet(button_style)
    info_button.clicked.connect(help_info)
    button.clicked.connect(search_files_window)

    layout.addWidget(info_button)
    layout.addWidget(button)
    window.setLayout(layout)
    window.show()


    app.exec_()
