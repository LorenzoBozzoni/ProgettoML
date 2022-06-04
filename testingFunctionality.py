
from asyncio.windows_events import NULL
from calendar import EPOCH
from cmath import nan
from math import floor, sqrt
import random as rn
import time
from matplotlib import projections
from matplotlib.ft2font import HORIZONTAL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dtreeviz.trees import dtreeviz # remember to load the package
from mpl_toolkits import mplot3d
from sklearn import linear_model
import sklearn
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import warnings
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
# not showing warnings in terminal window
warnings.filterwarnings("ignore")

# development
model = models.Sequential()
model.add(layers.Dense(60, activation="relu", input_shape=(X_train.shape[1],)))
model.add(layers.Dense(15, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=['BinaryAccuracy']
)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# ten numbers with equal distance from 0 to 50 (5,10,15,20,...)
EPOCH_NUMBER = np.linspace(1, 12, 7)    #5
# ten numbers with equal distance from 0 to 50 (5,10,15,20,...)
BATCH_SIZE = np.linspace(100, 50, 4) #4


#print('BATCH_SIZE',BATCH_SIZE,'EPOCH_NUMBER',EPOCH_NUMBER)



column_names = ["EpochNumber", "BatchSize", "Precision","Loss"]
dfToScatter = pd.DataFrame(columns = column_names)

for e_element in EPOCH_NUMBER:
    for b_element in BATCH_SIZE:
        # Fit the model to the training data and record events into a History object.
        history = model.fit(
            X_train,
            y_train,
            epochs=int(e_element),
            batch_size=int(b_element),
            validation_split=0.2,
            verbose=1,
        )
        # Model evaluation
        test_loss, test_pr = model.evaluate(X_test, y_test)
        print(test_pr)

        dfToAppend = pd.DataFrame([[e_element, b_element, test_pr, test_loss]]  , columns=column_names)
        dfToScatter = dfToScatter.append(dfToAppend)
        dfToAppend = dfToAppend[0:0]

fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(dfToScatter["EpochNumber"], dfToScatter["BatchSize"], dfToScatter["Precision"], edgecolors="red",linewidths=6)
ax.set_title('Precision')
ax.set_xlabel("Epoch number")
ax.set_ylabel("Batch size")
ax.set_zlabel("Precision")
print("x:",dfToScatter["EpochNumber"])   
print("y:",dfToScatter["BatchSize"])    
print("z:",dfToScatter["Precision"])
print("colonne",list(dfToScatter.columns))


ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(dfToScatter["EpochNumber"], dfToScatter["BatchSize"], dfToScatter["Loss"], edgecolors="red",linewidths=6)
ax2.set_title('Loss')
ax2.set_xlabel("Epoch number")
ax2.set_ylabel("Batch size")
ax2.set_zlabel("Loss")
print("x:",dfToScatter["EpochNumber"])   
print("y:",dfToScatter["BatchSize"])    
print("z:",dfToScatter["Precision"])
plt.show()

print(dfToScatter)















'''
EPOCH_NUMBER = 100
BATCH_SIZE = 1000
# Fit the model to the training data and record events into a History object.
history = model.fit(X_train,y_train,epochs=EPOCH_NUMBER,batch_size=BATCH_SIZE,validation_split=0.2,verbose=1) #,callbacks=[es],batch_size=BATCH_SIZE
# Model evaluation
test_loss, test_pr = model.evaluate(X_test, y_test)
print(test_pr)

# Plot loss (y axis) and epochs (x axis) for training set and validation set
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.epoch, np.array(history.history["accuracy"]) * 100, label="Train accuracy")
plt.plot(history.epoch, np.array(history.history["val_accuracy"]) * 100, label="Val accuracy")
plt.plot(history.epoch, np.array(history.history["loss"]), label="Train lost")
plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Val lost")
plt.legend()
plt.show()

        # Plot loss (y axis) and epochs (x axis) for training set and validation set
        # plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Parameters")

        plt.plot(history.epoch, np.array(history.history["loss"]), label="Train loss")
        plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Val loss")
        plt.plot(history.epoch, np.array(history.history["accuracy"]) * 100, label="Train accuracy")
        plt.plot(history.epoch, np.array(history.history["val_accuracy"]) * 100, label="Val accuracy")
        plt.legend()
    plt.show()
'''




