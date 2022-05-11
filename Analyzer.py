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


df = pd.read_csv("./Loan_Default.csv")
# df.drop(["ID", "year"], axis=1, inplace=True)

target = "Status"
labels = ["Defaulter", "Not-Defaulter"]
features = [i for i in df.columns.values if i not in [target]]

original_df = df.copy(deep=True)

""" @deprecated - Useless things used in first instance
print(
    "\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.".format(
        df.shape[1], df.shape[0]
    )
)




df.info()

ID_ARRAY = np.array(df[["ID"]])

print("TOTAL LENGTH:", len(df))
print("FIRST ELEMENT: ", ID_ARRAY[0])
print("LAST ELEMENT: ", ID_ARRAY[len(ID_ARRAY) - 1])
print("DIFFERENCE: ", ID_ARRAY[len(ID_ARRAY) - 1] - ID_ARRAY[0])


y = df.pop("Status")
X = df
bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns
print(featureScores)"""

df.drop(["ID", "year"], axis=1, inplace=True)
s = df["Status"]
# display(df.describe())


"""3D PLOT EXAMPLE

fig = plt.figure()
ax = plt.axes(projection="3d")

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, "gray")

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Greens")
plt.show()
"""


cat_features = []
num_features = []

fig, ax = plt.subplots(6, 6)
coordx = 0
coordy = 0

# DATA VISUALIZATION
for column in df.columns:
    # print("coordx", coordx, "coordy", coordy)
    if coordx == 4 and coordy == 6:
        break
    elif coordx == 6:
        coordx = 0
        coordy += 1

    frequency = df[column].value_counts(dropna=False)  #

    """ @debug - Printing categories and their relative cardinality 
    print(
        "frequency.index of lenght",
        frequency.index.values.size,
        " : ",
        frequency.index.values,
        " frequency.values: of lenght",
        frequency.values.size,
        " : ",
        frequency.values,
    )"""
    # plt.figure(figsize=(10, 10))

    """ GET THE LABEL FOR NUMBER OF OCCURRENCES
    print(type(frequency.index.values.tolist()))
    """

    # plt.subplot(1, 2, 1)
    if df[column].nunique() <= 7:
        cat_features.append(df[column])
        sns.barplot(
            x=frequency.index, y=frequency.values, alpha=0.8, ax=ax[coordx, coordy]
        )
        coordx += 1
    else:
        num_features.append(df[column])
        maxF = max(frequency.values)
        minF = min(frequency.values)
        barWidth = (maxF - minF) / 15
        sns.set_style("whitegrid")
        sns.distplot(
            df[column].dropna(),
            kde=False,
            color="red",
            bins=15,
            ax=ax[coordx, coordy],
        )
        coordx += 1
    # sns.barplot
    # sns.pointplot

    plt.title("Distribution of values for column " + column)
    plt.ylabel("Number of Occurrences", fontsize=12)
    plt.xlabel("Values", fontsize=12)

    # valueNan = frequency.index.values[2]
    # print(type(valueNan))

    """ @deprecated - DIAGRAMMA A TORTA PER VALORI NULLI
    nnv = 0
    nv = 0
    for i in range(0, len(frequency.values)):
        if not pd.isna(frequency.index.values[i]):
            nnv += frequency.values[i]
        else:
            nv += frequency.values[i]

    pielist = [(nnv / (nnv + nv)), (nv / (nnv + nv))]
    colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.9, len(pielist)))
    print(pielist)

    
    print(
        "frequency.index of lenght",
        frequency.index.values.size,
        " : ",
        frequency.index.values,
        " frequency.values: of lenght",
        frequency.values.size,
        " : ",
        frequency.values,
    )
    plt.subplot(1, 2, 2)
    plt.pie(
        pielist,  # frequency.values,
        colors=colors,
        radius=3,
        center=(4, 4),
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        frame=True,
    )"""


plt.show()


# questa riga sotto è corretta ma usa uno spatasso di RAM
# sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", robust=True)
sns.heatmap(
    df[["age", "Interest_rate_spread", "rate_of_interest"]].isnull(), cmap="viridis"
)


plt.show()


dfNumeric = df  # copy of the dataframe, df contains categorical values while dfNumeric has all those values converted in numeric

# substituing categorical values with numerical ones
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() <= 7 and columnName != "Status":
        print("columnName:", columnName)
        freqSeries = columnData.value_counts()
        # print("PRIMA:", columnData)
        # print("columnData.value_counts().values: ", columnData.value_counts().index)
        # print("np.arange(0, columnData.nunique())", np.arange(0, columnData.nunique()))
        dfNumeric[columnName].replace(
            columnData.value_counts().index,
            np.arange(0, columnData.nunique()),
            inplace=True,
        )
        # print("DOPO:", df[columnName])

pd.set_option(
    "display.max_columns", None
)  # used when displaying informations regarding an high number of columns (by default some are omitted)


# displaying information for the newly created dataframe
display(dfNumeric.describe(include="all"))


# plotting correlation grid for all features (used for null values management)
fig, ax = plt.subplots(figsize=(32, 32))
ax = sns.heatmap(
    dfNumeric.corr(), vmin=-1, vmax=1, cmap="BrBG"
)  # non prende le features categoriche
plt.show()
