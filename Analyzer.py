from asyncio.windows_events import NULL
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, f1_score, confusion_matrix
import seaborn as sns
import warnings
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
# not showing warnings in terminal window
warnings.filterwarnings("ignore")


# 10 FEATURES NUMERICHE
# 22 FEATURES CATEGORICHE

#region functions
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
            axd[letters[indexer]].set_title(columnName)
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
            axd[letters[indexer]].set_title(columnName)           
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
#endregion



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


#region removing outvalues
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() > 7 and columnName != "Status":

        max_thresold = dfNumeric[columnName].quantile(0.80)
        min_thresold = dfNumeric[columnName].quantile(0.20)
        print("COLONNA CON OUTVALUES", columnName, "MAXPERC",max_thresold, "MINPERC",min_thresold)
        dfNumeric = df[(df[columnName] < max_thresold) & (df[columnName] > min_thresold)]
#endregion

#region substituing categorical values with numerical ones
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() <= 7 and columnName != "Status":
        # print("columnName:", columnName)
        freqSeries = columnData.value_counts()
        #print("columnData.value_counts().index", columnData.value_counts().index)
        #print("np.arange(0, columnData.nunique())",np.arange(0, columnData.nunique()))

        dfNumeric[columnName].replace(columnData.value_counts().index,np.arange(0, columnData.nunique()),inplace=True)
        #print(dfNumeric[columnName])
#endregion



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
dfNumeric = dfNumeric.drop(["construction_type","Secured_by","Security_Type"], axis=1)       # feature selection


#visualizeNumerical(dfNumeric)




X = dfNumeric.drop(target, axis = 1)
y = dfNumeric[target]



'''  SAVING dfNumeric IN CSV FILE 
import os  
os.makedirs('./', exist_ok=True)  
dfNumeric.to_csv('./out.csv') 

'''
# setting random seed for randomsearch for hyperparameters
rn.seed(time.process_time())

# dividing in training and test set for using kfold cross validation in training set in order to find best parameters 
# not using default split function because it returns matrices and vector, and we want instead dataframe 
X_train = X[1:floor(X.shape[0] * 0.8)]
X_test = X[floor((X.shape[0] * 0.8) + 1):]
y_train =  y[1:floor(len(y) * 0.8)]
y_test = y[floor((len(y) * 0.8) + 1):]


# obtaining matrices size
print("ShapeDf",df.shape)
print("ShapeDfNumeric",dfNumeric.shape)
print("ShapeX",X.shape)
print("Shapey",y.shape)
print("ShapeX_train",X_train.shape)
print("ShapeX_test",X_test.shape)
print("ShapeXìy_train",y_train.shape)
print("Shapey_test",y_test.shape)

print("Full X_train", X_train, "type: ", type(X_train))


# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print("Size X_scaled:", X_scaled.shape)

# dimensionality reduction
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
print("Size X_PCA:", X_pca.shape)




 

v_param_index = 0

best_parameters = [[0,0,0],[0,0,0],[0,0,0]]



#region LogisticRegression
# logistic regression has not parameters to be tuned
reg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
scores = cross_val_score(reg, X_pca,y_train, cv=5, scoring="accuracy")                           # scores è un vettore numpy quindi bisogna vedere quello
print(scores.mean())
#endregion


'''
#region DecisionTreeClassifier
# indipendente da min_sample_leaf per valori bassi (100-1000), nel range (2000-10000) valore costanti 0.977554
clf = DecisionTreeClassifier(class_weight='balanced') #min_samples_leaf=5000, max_depth=10
sample_range = list(np.arange(1000,5001,1000)) #15001
depth_range = list(range(2,8))
param_grid = dict(min_samples_leaf=sample_range, max_depth=depth_range)    
print(param_grid)
grid = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring="accuracy")    #RandomizedSearchCV , random_state=rn.randint(0,10)
grid.fit(X_pca,y_train)
score_df = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("DecisionTree:",score_df)
print(grid.best_score_)
print(grid.best_params_)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
parameters = str(score_df["params"].values)

print("parameters",parameters)

xy_list = parameters [:-1].split("\n")

x = np.empty(len(xy_list))
y = np.empty(len(xy_list))

counter = 0
for element in xy_list:
    x[counter] = element[:-1].split(", ")[0].split(":")[1]
    y[counter] = element[:-1].split(", ")[1].split(":")[1]
    counter += 1

ax.scatter(x, y, score_df["mean_test_score"], edgecolors="green",linewidths=6)
ax.set_title('Decision Tree Classifier')
ax.set_xlabel("max_depth")
ax.set_ylabel("min_sample_leaf")
ax.set_zlabel("mean_test_score")
print("x:",score_df["params"])    # get("max_depth")
print("y:",type(score_df["params"].values))     # .get("min_sample_leaf")
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()
best_parameters[v_param_index][0] = "DecisionTreeClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion





v_param_index += 1






#region AdaBoostClassifier
modelAda = AdaBoostClassifier()   #n_estimators = 2
estimators_range = list(np.arange(1,16,2))
learning_rate_range = list(np.arange(0.1,2.1,0.1))
param_grid = dict(n_estimators=estimators_range, learning_rate=learning_rate_range)    
print(param_grid)
grid = GridSearchCV(modelAda, param_grid=param_grid, cv=5, scoring="accuracy")
grid.fit(X_pca,y_train)
score_df = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("AdaBoost:",score_df)
print(grid.best_score_)
print(grid.best_params_)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
parameters = str(score_df["params"].values)

print("parameters",parameters)

xy_list = parameters [:-1].split("\n")

x = np.empty(len(xy_list))
y = np.empty(len(xy_list))

counter = 0
for element in xy_list:
    x[counter] = element[:-1].split(", ")[0].split(":")[1]
    y[counter] = element[:-1].split(", ")[1].split(":")[1]
    counter += 1

ax.scatter(x, y, score_df["mean_test_score"], edgecolors="blue",linewidths=6)
ax.set_title('adaboost classifier score')
ax.set_xlabel("estimators number")
ax.set_ylabel("learning rate")
ax.set_zlabel("mean_test_score")
print("x:",score_df["params"])    # get("max_depth")
print("y:",type(score_df["params"].values))     # .get("min_sample_leaf")
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()

best_parameters[v_param_index][0] = "AdaBoostClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion



v_param_index += 1



#region GradientBoostingClassifier
modelGrad= GradientBoostingClassifier()
estimators_range = list(np.arange(1,16,2))
learning_rate_range = list(np.arange(0.1,2.1,0.1))
param_grid = dict(n_estimators=estimators_range, learning_rate=learning_rate_range)    
print(param_grid)
grid = GridSearchCV(modelGrad, param_grid=param_grid, cv=5, scoring="accuracy")
grid.fit(X_pca,y_train)
score_df = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("GradientBoosting:",score_df)
print(grid.best_score_)
print(grid.best_params_)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
parameters = str(score_df["params"].values)

print("parameters",parameters)

xy_list = parameters [:-1].split("\n")

x = np.empty(len(xy_list))
y = np.empty(len(xy_list))

counter = 0
for element in xy_list:
    x[counter] = element[:-1].split(", ")[0].split(":")[1]
    y[counter] = element[:-1].split(", ")[1].split(":")[1]
    counter += 1

ax.scatter(x, y, score_df["mean_test_score"], edgecolors="red",linewidths=6)
ax.set_title('gradient boosting score')
ax.set_xlabel("estimators number")
ax.set_ylabel("learning rate")
ax.set_zlabel("mean_test_score")
print("x:",score_df["params"])    # get("max_depth")
print("y:",type(score_df["params"].values))     # .get("min_sample_leaf")
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()


best_parameters[v_param_index][0] = "GradientBoostingClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion





'''

#region BaggingClassifier
modelBagging = BaggingClassifier(base_estimator=reg)
estimators_range = list(np.arange(1,30,2))
bootstrap_range = list([True,False])
param_grid = dict(n_estimators=estimators_range, bootstrap=bootstrap_range)    
print(param_grid)
grid = GridSearchCV(modelBagging, param_grid=param_grid, cv=5, scoring="accuracy")
grid.fit(X_pca,y_train)
score_df = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("BaggingClassifier:",score_df)
print(grid.best_score_)
print(grid.best_params_)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
parameters = str(score_df["params"].values)

print("parameters",parameters)

xy_list = parameters [:-1].split("\n")

x = [None] * (len(xy_list))
y = np.empty(len(xy_list))

counter = 0
for element in xy_list:
    x[counter] = element[:-1].split(", ")[0].split(":")[1][1:]
    y[counter] = element[:-1].split(", ")[1].split(":")[1]
    counter += 1

print("x prima",x)
for i in range(0,len(x)): 
    if x[i] == "False":
        x[i] = 0
    else:
        x[i] = 1 
    i += 1 

print("x dopo",x)
ax.scatter(x, y, score_df["mean_test_score"], edgecolors="red",linewidths=6)
ax.set_title('boosting classifier score')
ax.set_xlabel("bootstrap")
ax.set_ylabel("estimators number")
ax.set_zlabel("mean_test_score")
print("x:",score_df["params"])    # get("max_depth")
print("y:",type(score_df["params"].values))     # .get("min_sample_leaf")
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()


best_parameters[v_param_index][0] = "GradientBoostingClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion




print(best_parameters)





''' Parte successiva da implementare
overall_score = []
for element in best_parameters:
    if element[0] == "AdaBoostClassifier":
        
        ada = AdaBoostClassifier(learning_rate=element[1].get("learning_rate"))
        ada.fit(X_test)
        y_pred = ada.predict(y_test)

        #e qui ci mettiamo una bella confusion matrix


'''



















 







'''

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators = 1000)
model.fit(X_train,y_train)


'''







