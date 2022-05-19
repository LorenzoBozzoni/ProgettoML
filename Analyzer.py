from asyncio.windows_events import NULL
from cmath import nan
from math import sqrt
from turtle import title
from unittest import TextTestRunner
from joblib import PrintTime
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
from IPython.display import display
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, f1_score, confusion_matrix
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
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

        max_thresold = dfNumeric[columnName].quantile(0.995)
        min_thresold = dfNumeric[columnName].quantile(0.005)
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

#visualizeNumerical(dfNumeric)




X = dfNumeric.drop(target, axis = 1)
y = dfNumeric[target]


''' obtaining matrices size
print("ShapeDf",df.shape)
print("ShapeDfNumeric",dfNumeric.shape)

print("ShapeX",X.shape)
print("Shapey",y.shape)
'''

'''  SAVING dfNumeric IN CSV FILE 
import os  
os.makedirs('./', exist_ok=True)  
dfNumeric.to_csv('./out.csv') 

'''




# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Size X_scaled:", X_scaled.shape)

# dimensionality reduction
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
print("Size X_PCA:", X_pca.shape)


# logistic regression has not parameters to be tuned
reg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
scores = cross_val_score(reg, X_pca,y, cv=10, scoring="accuracy")                           # scores è un vettore numpy quindi bisogna vedere quello
print(scores.mean())


# indipendente da min_sample_leaf per valori bassi (100-1000), nel range (2000-10000) valore costanti 0.977554
clf = DecisionTreeClassifier(class_weight='balanced') #min_samples_leaf=5000, max_depth=10
sample_range = list(np.arange(10000,15001,1000))
depth_range = list(range(2,8))
param_grid = dict(min_samples_leaf=sample_range, max_depth=depth_range)    
print(param_grid)
grid = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring="accuracy")
grid.fit(X_pca,y)
santocielo = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("DecisionTree:",santocielo)
#print(grid.best_score_)
#print(grid.best_params_)



modelAda = AdaBoostClassifier()   #n_estimators = 2
estimators_range = list(np.arange(50,76,5))
learning_rate_range = list(np.arange(4,6,1))
param_grid = dict(n_estimators=estimators_range, learning_rate=learning_rate_range)    
print(param_grid)
grid = GridSearchCV(modelAda, param_grid=param_grid, cv=5, scoring="accuracy")
grid.fit(X_pca,y)
santocielo = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print("AdaBoost:",santocielo)
#endregion


































'''
# dividing values in train and test part
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

kf =KFold(n_splits=10, random_state=None, shuffle=False) #initialize KFold
for train_index, validation_index in kf.split(X):
    print("TRAIN:", train_index, "VALIDATION:", validation_index)
    X_train = X.iloc[train_index]
    X_validation = X.iloc[validation_index]
    y_train= y[train_index]
    y_validation = y[validation_index]



#region linear regression model
regr = linear_model.LinearRegression()  # creating a linear model object
regr.fit(X_train, y_train)
y_pred_regr = regr.predict(X_test)
print('Mean square error: %.2f' % mean_squared_error(y_test, y_pred_regr))
#endregion



#region building a logistic regression model
logreg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(logreg.score(X_test,y_test))
# visualize model score (for classification tasks)
getScoreMetrics(y_test=y_test, y_pred=y_pred_logreg)
#endregion



#region building a decision tree classifier model
clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced',min_samples_leaf=5000)   
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
getScoreMetrics(y_test=y_test, y_pred=y_pred_clf)
plot_tree(clf, filled=True)
plt.show()
# another visualization for decision tree classifier
viz = dtreeviz(clf, X, y,target_name="Status",feature_names=dfNumeric.columns)
viz.view()
#endregion



#region polynomial linear regression
poly = PolynomialFeatures(degree=1)
X_poly= poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.33)
# Create linear regression object
regr = linear_model.Ridge(alpha=5.)#LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)
y_train_pred = regr.predict(X_train)
# The mean squared error
#print('Mean squared error on train: %.2f'% mean_squared_error(y_test, y_pred))
rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
print('Mean squared error on train: %.2f'% mean_squared_error(y_train, y_train_pred))
# The mean squared error
print('Mean squared error on test: %.2f'% mean_squared_error(y_test, y_pred))
rmse_test = sqrt(mean_squared_error(y_test, y_pred))
#endregion



#region AdaBoostClassifier model
modelAda= AdaBoostClassifier(n_estimators = 2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
modelAda.fit(X_train,y_train)
y_pred = modelAda.predict(X_test)
getScoreMetrics(y_test=y_test, y_pred=y_pred)
#endregion



#region GradientBoostingClassifier model
modelGrad= GradientBoostingClassifier(n_estimators = 2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
modelGrad.fit(X_train,y_train)
y_pred = modelGrad.predict(X_test)
getScoreMetrics(y_test=y_test, y_pred=y_pred)
#endregion




#region XGBClassifier model
modelXGB = XGBClassifier(n_estimators = 2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
modelXGB.fit(X_train, y_train)
y_pred = modelXGB.predict(X_test)
getScoreMetrics(y_test=y_test, y_pred=y_pred)
#endregion



from sklearn.ensemble import BaggingClassifier
model= BaggingClassifier(n_estimators = 200)
model.fit(X_train,y_train)


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators = 1000)
model.fit(X_train,y_train)


'''







