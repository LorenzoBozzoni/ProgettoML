from asyncio.windows_events import NULL
from cmath import nan
from turtle import title
from unittest import TextTestRunner
from joblib import PrintTime
from matplotlib.ft2font import HORIZONTAL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# 10 FEATURES NUMERICHE
# 22 FEATURES CATEGORICHE

# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

def visualizeCategorical(df):
    letters = ["A","B","C","D","E","F","G","H","I","L","M","N","O","P","Q","R","S","T","U","V","X","Y"]
    indexer = -1

    fig = plt.figure(constrained_layout=True)

    axd = fig.subplot_mosaic(
        """
                ABCDE
                FGHIL
                MNOPQ
                RSTUV
                XY...
                """
    )

    for (columnName, columnData) in df.iteritems():
        frequency = df[columnName].value_counts(dropna=False)  #
        if df[columnName].nunique() <= 7:
            times = df[columnName].value_counts()
            axd[letters[indexer]].bar(times.index, times.values)
            indexer += 1
    
    #identify_axes(axd)
    fig.show()
    plt.show()


def visualizeNumerical(df):
    letters = ["A","B","C","D","E","F","G","H","I","L"]
    indexer = -1

    fig = plt.figure(constrained_layout=True)

    axd = fig.subplot_mosaic(
        """
                ABCDE
                FGHIL
                """
    )

    for (columnName, columnData) in df.iteritems():
        frequency = df[columnName].value_counts(dropna=False)  #
        if df[columnName].nunique() > 7:
            times = df[columnName].value_counts()
            axd[letters[indexer]].hist(df[columnName],bins="auto")
            indexer += 1

    fig.show()
    plt.show()

def getScoreMetrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy: ", acc)
    rec = recall_score(y_test, y_pred)
    print("Recall: ", rec)
    f1score = 2 / ((1 / acc) + (1 / rec))
    print("f1 score calculated manually: ", f1score)
    print("f1 score calculated automatically: ", f1_score(y_test, y_pred))

    m = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )  # labels ti permette di specificare quale label (target output) considerare nella confusion matrix
    print(m)




pd.set_option("display.max_columns", None)  # used when displaying informations regarding an high number of columns (by default some are omitted)
plt.style.use("seaborn")
df = pd.read_csv("./Loan_Default.csv")
df.drop(["ID", "year"], axis=1, inplace=True)

target = "Status"
labels = ["Defaulter", "Not-Defaulter"]
features = [i for i in df.columns.values if i not in [target]]

original_df = df.copy(deep=True)


cat_features = []
num_features = []


#visualizeCategorical(df=df)
#visualizeNumerical(df=df)

# questa riga sotto è corretta ma usa uno spatasso di RAM
# sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", robust=True)

# vediamo la presenza di valori nulli nelle features con maggior numero
"""sns.heatmap(df[["age", "Interest_rate_spread", "rate_of_interest"]].isnull(), cmap="viridis")
plt.show()"""


dfNumeric = df  # copy of the dataframe, df contains categorical values while dfNumeric has all those values converted in numeric


# removing outvalues
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() > 7 and columnName != "Status":

        max_thresold = dfNumeric[columnName].quantile(0.995)
        min_thresold = dfNumeric[columnName].quantile(0.005)
        print("COLONNA CON OUTVALUES", columnName, "MAXPERC",max_thresold, "MINPERC",min_thresold)
        dfNumeric = df[(df[columnName] < max_thresold) & (df[columnName] > min_thresold)]

# substituing categorical values with numerical ones
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() <= 7 and columnName != "Status":
        # print("columnName:", columnName)
        freqSeries = columnData.value_counts()
        #print("columnData.value_counts().index", columnData.value_counts().index)
        #print("np.arange(0, columnData.nunique())",np.arange(0, columnData.nunique()))

        dfNumeric[columnName].replace(columnData.value_counts().index,np.arange(0, columnData.nunique()),inplace=True)
        #print(dfNumeric[columnName])






# displaying information for the newly created dataframe
# display(dfNumeric.describe(include="all"))


# plotting correlation grid for all features (used for null values management)
fig, ax = plt.subplots(figsize=(32, 32))
ax = sns.heatmap(dfNumeric.corr(), vmin=-1, vmax=1, cmap="YlGnBu")  # non prende le features categoriche. Per avere i valori: "annot=True"
plt.show()


#visualizeNumerical(dfNumeric)     # visualize numeric feature's histograms after removing outliers

#print(dfNumeric.corr())
#print(dfNumeric.isnull().sum())


dfNumeric.fillna(dfNumeric.mean(),inplace=True)

#visualizeNumerical(dfNumeric)




X = dfNumeric.drop(target, axis = 1)
y = dfNumeric[target]


'''print("ShapeDf",df.shape)
print("ShapeDfNumeric",dfNumeric.shape)

print("ShapeX",X.shape)
print("Shapey",y.shape)'''


import os  
os.makedirs('./', exist_ok=True)  
dfNumeric.to_csv('./out.csv') 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y)
'''regr = linear_model.LinearRegression()  # creating a linear model object
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
'''

logreg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(logreg.score(X_test,y_test))

getScoreMetrics(y_test=y_test, y_pred=y_pred)

#visualizeCategorical(df=dfNumeric)

print(y_test)
print(sum(y_test))