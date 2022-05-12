from asyncio.windows_events import NULL
from cmath import nan
from joblib import PrintTime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from IPython.display import display
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# 10 FEATURES NUMERICHE
# 22 FEATURES CATEGORICHE


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





plt.style.use("seaborn")
df = pd.read_csv("./Loan_Default.csv")
df.drop(["ID", "year"], axis=1, inplace=True)

target = "Status"
labels = ["Defaulter", "Not-Defaulter"]
features = [i for i in df.columns.values if i not in [target]]

original_df = df.copy(deep=True)


cat_features = []
num_features = []


visualizeCategorical(df=df)
visualizeNumerical(df=df)

# questa riga sotto è corretta ma usa uno spatasso di RAM
# sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", robust=True)

# vediamo la presenza di valori nulli nelle features con maggior numero
"""sns.heatmap(df[["age", "Interest_rate_spread", "rate_of_interest"]].isnull(), cmap="viridis")
plt.show()"""


dfNumeric = df  # copy of the dataframe, df contains categorical values while dfNumeric has all those values converted in numeric

# substituing categorical values with numerical ones
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() <= 7 and columnName != "Status":
        # print("columnName:", columnName)
        freqSeries = columnData.value_counts()
        dfNumeric[columnName].replace(columnData.value_counts().index,np.arange(0, columnData.nunique()),inplace=True)
    else:
        max_thresold = df[columnName].quantile(0.995)
        min_thresold = df[columnName].quantile(0.005)
        dfNumeric = df[
            (df[columnName] < max_thresold) & (df[columnName] > min_thresold)
        ]


pd.set_option("display.max_columns", None)  # used when displaying informations regarding an high number of columns (by default some are omitted)


# displaying information for the newly created dataframe
# display(dfNumeric.describe(include="all"))


# plotting correlation grid for all features (used for null values management)
fig, ax = plt.subplots(figsize=(32, 32))
ax = sns.heatmap(dfNumeric.corr(), vmin=-1, vmax=1, cmap="BrBG")  # non prende le features categoriche
plt.show()


visualizeNumerical(dfNumeric)
