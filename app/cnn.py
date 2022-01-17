import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

import shutil
import os
import random
import keras.backend as K
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator

# ,kernel_initializer = glorot_uniform(seed=9)
def GenreModel(input_shape = (288,432,4),classes=9):

  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(2,2))(X_input)
  X = Activation('relu')(X)
  X = BatchNormalization(axis=3)(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (2,2))(X)
  X = Activation('relu')(X)
  X = BatchNormalization(axis=3)(X)

  X = Conv2D(32,kernel_size=(3,3),strides = (2,2))(X)
  X = Activation('relu')(X)
  X = BatchNormalization(axis=3)(X)

  X = Flatten()(X)
#   X = Dropout(rate=0.2)(X)


  X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model


#from old keras source code
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



def main():
    # genres = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae','rock']

    # directory = "./spectrograms3sec"
    # destination = shutil.copytree(directory, "./spectrograms3sec_train", copy_function = shutil.copy) 

    # for g in genres:

    #     path = os.path.join('./spectrograms3sec_test/',f'{g}')
    #     os. makedirs(path)

    # for g in genres:
    #     filenames = os.listdir(os.path.join(destination,f"{g}"))
    #     random.shuffle(filenames)
    #     test_files = filenames[0:round(len(filenames)*0.2)]

    # for f in test_files:

    #     shutil.move(destination + "/"+f"{g}"+ "/" + f,"./spectrograms3sec_test/" + f"{g}")

    train_dir = "./spectrograms3sec_train/"
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

    validation_dir = "./spectrograms3sec_test/"
    vali_datagen = ImageDataGenerator(rescale=1./255)
    vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

    model = GenreModel(input_shape=(288,432,4),classes=9)

    opt = SGD(learning_rate=0.0005, decay=1e-6)
    model.compile(optimizer = opt,loss='categorical_crossentropy',metrics=['categorical_accuracy',get_f1])
    model.summary()

    model.fit(train_generator,epochs=30,validation_data=vali_generator)
    # model.save('./finalized_model_9')



if __name__ == '__main__':
    main()