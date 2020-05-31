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

# Model evaluation
start_time = time.time()

model = load_model('cnn_ecgview.h5')

y_pred = model.predict(x_cnn_test)
y_pred = np.argmax(y_pred, axis=-1)

print("--- inference time of %s seconds ---" % (time.time() - start_time))

# Get F1 score
f1 = f1_score(y_cnn_test, y_pred, average="macro")
print("Test f1 score : %s "% f1)

# Get ROC AUC
roc = roc_auc_score(y_cnn_test, y_pred)
print("Test ROC AUC score : %s "% roc)

# Get the accuracy
acc = accuracy_score(y_cnn_test, y_pred)
print("Test accuracy score : %s "% acc)

# Get the specificity
tn, fp, fn, tp = confusion_matrix(y_cnn_test, y_pred).ravel()
specificity = tn / (tn+fp)
print("Specificity : %s "% specificity)

# Get the sensitivity
sensitivity= tp / (tp+fn)
print("Sensitivity: %s "% sensitivity)
