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


df = pd.read_csv("./Loan_Default.csv")
# df.drop(["ID", "year"], axis=1, inplace=True)

target = "Status"
labels = ["Defaulter", "Not-Defaulter"]
features = [i for i in df.columns.values if i not in [target]]

original_df = df.copy(deep=True)

"""
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
"""

"""y = df.pop("Status")
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
display(df.describe())


"""a = np.array(df[["ID"]])
b = np.array(df[["Upfront_charges"]])
c = np.array(df[["Interest_rate_spread"]])
plt.plot(a, b, "rd")
plt.show()
print(a)


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(c, b, a, "gray")"""


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


""" PLOT MULTIPLE GRAPH ALTOGETHER
figure, axis = plt.subplots(4, 4)

for i in range(0,4):
    for j in range(0,4):
        axis[i,j].bar(qualitative_features[i*j + j],  )"""


cat_features = []
num_features = []

fig, ax = plt.subplots(6, 6)
coordx = 0
coordy = 0

# DATA VISUALIZATION
for column in df.columns:
    print("coordx", coordx, "coordy", coordy)
    if coordx == 4 and coordy == 6:
        break
    elif coordx == 6:
        coordx = 0
        coordy += 1

    frequency = df[column].value_counts(dropna=False)  #

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
            df[column],
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

    """    nnv = 0
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


fig.show()
plt.show()


"""for index in range(0, len(cat_features)):
    values = df[:, index]
    print("TYPE", type(values))

    # for element in column:"""


# dummies = pd.get_dummies(df, columns=["age"])
# print(dummies.head)

fig, ax = plt.subplots()
ax.set_title("Null values")
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", robust=True)
fig.show()
plt.show()
