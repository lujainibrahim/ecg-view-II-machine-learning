# Import libraries
import pandas as pd
import numpy as np
import pylab as plt
import imblearn
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from keras import optimizers, losses, activations, models, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from keras.utils import to_categorical
from keras.models import load_model, Sequential
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
from collections import Counter
import random
import time

# Load data
# Import train and test data into dataframes from csv files produced using the data processing code
df_cnn_train = pd.read_csv("train.csv", header=None)
df_cnn_train = df_cnn_train.sample(frac=1)
df_cnn_test = pd.read_csv("test.csv", header=None)

# Get data from dataframes
y_cnn = np.array(df_cnn_train[11].values).astype(np.int8)
y_cnn=to_categorical(y_cnn)
x_cnn = np.array(df_cnn_train[list(range(11))].values)[..., np.newaxis]
y_cnn_test = np.array(df_cnn_test[11].values).astype(np.int8)
x_cnn_test = np.array(df_cnn_test[list(range(11))].values)[..., np.newaxis]

# Model definition
def get_model(learning_rate=0.001):
    nclass = 2
    inp = Input(shape=(11, 1))

    cnn = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    cnn = Dropout(rate=0.1)(cnn)
    cnn = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(cnn)
    cnn = Dropout(rate=0.1)(cnn)
    cnn = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(cnn)
    cnn = Dropout(rate=0.1)(cnn)
    cnn = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(cnn)
    cnn = GlobalMaxPool1D()(cnn)
    cnn = Dropout(rate=0.1)(cnn)
    dense_1 = Dense(64, activation=activations.relu, name="dense_1", kernel_regularizer=regularizers.l2(l=0.1))(cnn)
    dense_1 = Dense(16, activation=activations.relu, name="dense_2", kernel_regularizer=regularizers.l2(l=0.1))(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_ecg_view")(dense_1)


    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(learning_rate)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    return model

# Model Training
# Load model and model summary
model = get_model()
model.summary()

# File path to save the model
file_path = "cnn_ecgview.h5"

# Checkpoint the model's weight based on the accuracy of the model
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Set early stopping based on accuracy. It stops after 10 consecutive epochs of no accuracy improvement.
early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)

# Reduce learning rate based on accuracy. It reduces the rate after 7 consecutive epochs of no accuracy improvement.
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=7, verbose=2)

callbacks_list = [checkpoint, early, redonplat]

# Train the model, load weights into above file path to save the model
model.fit(x_cnn, y_cnn, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

# The file will be saved in the file_path and can be loaded later using Keras for evaluation
