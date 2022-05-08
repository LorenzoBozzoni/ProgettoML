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


"""
3D PLOT EXAMPLE

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

# istogramma con asse x i nomi delle features e y percentuale valori mancanti per quella features

# print(df.head)


"""figure, axis = plt.subplots(4, 4)

for i in range(0,4):
    for j in range(0,4):
        axis[i,j].bar(qualitative_features[i*j + j],  )"""


for column in df.columns:
    # numberOfValues = df[column].nunique()
    # differentValues = df[column].unique()
    # print("COLUMN NAME:",column," - NUMBER OF VALUES:",numberOfValues," - VALUES:",differentValues,"\n",)
    frequency = df[column].value_counts()  # dropna=False
    # plt.figure(figsize=(10, 5))
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
    plt.figure(figsize=(10, 5))

    print(type(frequency.index.values.tolist()))
    sns.histplot(x=frequency.index, y=frequency.values, alpha=0.8, stat="density")
    # sns.barplot
    # sns.pointplot

    # plt.title("Distribution of values for column", column)
    plt.ylabel("Number of Occurrences", fontsize=12)
    plt.xlabel("Values", fontsize=12)
    plt.show()

    # print(frequency)
    # f_matrix = np.array(frequency)
    # print("colonna 0:", f_matrix[:0], "colonna 1:", f_matrix[:1])
    # matrix = np.matrix(frequency)
    # plt.bar()
    # plt.plot(np.arange(0, numberOfValues, 1), differentValues, "r")
    # print("PRIMA COLONNA:", frequency[[]])
    #

    """plt.bar(
        frequency.index.values,
        frequency.values,
        width=1,
        edgecolor="white",
        linewidth=0.7,
    )"""

    # print("matrix column 0:", matrix[:0], "matrix column 1:", matrix[:1])
    # plt.plot(frequency.index.values, frequency.values, "r")
    # plt.show()


# qualitative_features.plot(kind="hist")
# plt.show()


# for c in df.columns.values:
#    print("nvalues", c, ": ", df[c].nunique())
