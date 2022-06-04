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


# 10 FEATURES NUMERICHE
# 22 FEATURES CATEGORICHE

#region functions

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


def getScoreMetrics(y_test, y_pred, modelName):
    acc = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1score =  f1_score(y_test, y_pred)
    f1scoreman = 2 / ((1 / acc) + (1 / rec))

    print("\nAccuracy: ", acc)
    print("Recall: ", rec)
    print("f1 score calculated manually: ", f1score)
    print("f1 score calculated automatically: ", f1scoreman)

    m = confusion_matrix(y_test, y_pred, labels=[0, 1])  # labels ti permette di specificare quale label (target output) considerare nella confusion matrix
    print(m)

    disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion matrix for " + modelName)
    plt.show()

    return [acc, rec, f1score, f1scoreman, modelName]
    
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
#sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", robust=True)
#plt.show()

# vediamo la presenza di valori nulli nelle features con maggior numero
"""sns.heatmap(df[["age", "Interest_rate_spread", "rate_of_interest"]].isnull(), cmap="viridis")
plt.show()"""


dfNumeric = df  # copy of the dataframe, df contains categorical values while dfNumeric has all those values converted in numeric


# removing outvalues
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() > 7 and columnName != "Status":
        max_thresold = dfNumeric[columnName].quantile(0.995)
        min_thresold = dfNumeric[columnName].quantile(0.005)
        dfNumeric = df[(df[columnName] < max_thresold) & (df[columnName] > min_thresold)]


# substituing categorical values with numerical ones
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() <= 7 and columnName != "Status":
        freqSeries = columnData.value_counts()
        dfNumeric[columnName].replace(columnData.value_counts().index,np.arange(0, columnData.nunique()),inplace=True)




# displaying information for the newly created dataframe
# display(dfNumeric.describe(include="all"))


# plotting correlation grid for all features (used for null values management)
#fig, ax = plt.subplots(figsize=(32, 32))
#ax = sns.heatmap(dfNumeric.corr(), vmin=-1, vmax=1, cmap="YlGnBu")  # non prende le features categoriche. Per avere i valori: "annot=True"
#plt.show()


# visualize numeric feature's histograms after removing outliers
#visualizeNumerical(dfNumeric)     

#print(dfNumeric.corr())
#print(dfNumeric.isnull().sum())

# replacing null values with the mean of the respective feature
dfNumeric.fillna(dfNumeric.mean(),inplace=True)

# removing features with no statistical contribution 
dfNumeric = dfNumeric.drop(["construction_type","Secured_by","Security_Type"], axis=1)       # feature selection


#visualizeNumerical(dfNumeric)



# dividing training features from target 
X = dfNumeric.drop(target, axis = 1)
y = dfNumeric[target]



'''  SAVING dfNumeric IN CSV FILE 
import os  
os.makedirs('./', exist_ok=True)  
dfNumeric.to_csv('./out.csv') 

'''
# setting random seed for randomsearch for hyperparameters
rn.seed(time.process_time())

sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

# dividing in training and test set for using kfold cross validation in training set in order to find best parameters 

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_res,y_res, test_size=0.20, stratify=y_res)

# convert the matrices into dataframes for using methods
X_train = pd.DataFrame(X_train_v, columns=X.columns)
X_test = pd.DataFrame(X_test_v, columns=X.columns)
y_train = pd.DataFrame(y_train_v, columns=["Status"])
y_test = pd.DataFrame(y_test_v, columns=["Status"])

# obtaining matrices size
print("ShapeDf",df.shape)
print("ShapeDfNumeric",dfNumeric.shape)
print("ShapeX",X.shape)
print("Shapey",y.shape)
print("ShapeX_train",X_train.shape)
print("ShapeX_test",X_test.shape)
print("Shapey_train",y_train.shape)
print("Shapey_test",y_test.shape)
#print("Full X_train", X_train, "type: ", type(X_train))


# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print("Size X_scaled:", X_scaled.shape)

# dimensionality reduction
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
print("Size X_PCA:", X_pca.shape)

import os  
os.makedirs('./', exist_ok=True)  
dfA = pd.DataFrame(X_pca) 
dfA.to_csv('./out.csv') 





'''
N = 10000

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

rndperm = np.random.permutation(dfNumeric.shape[0])
df_subset = df.loc[rndperm[:N],:].copy()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
 '''



bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] #naming the dataframe columns
print(featureScores)
print(bestfeatures.fit_transform(X,y))





v_param_index = 0
best_parameters = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
skf = StratifiedKFold(n_splits=10)


#region LogisticRegression
# logistic regression has not parameters to be tuned
reg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
scores = cross_val_score(reg, X_pca,y_train, cv=skf, scoring="balanced_accuracy")                           # scores è un vettore numpy quindi bisogna vedere quello
print(scores.mean())
#endregion

'''

#region DecisionTreeClassifier
# indipendente da min_sample_leaf per valori bassi (100-1000), nel range (2000-10000) valore costanti 0.977554
clf = DecisionTreeClassifier(class_weight='balanced') #min_samples_leaf=5000, max_depth=10
sample_range = list(np.arange(1000,5001,1000)) #15001
depth_range = list(range(2,10))
param_grid = dict(min_samples_leaf=sample_range, max_depth=depth_range)    
print(param_grid)
grid = GridSearchCV(clf, param_grid=param_grid, cv=skf, scoring="balanced_accuracy", verbose=3)    #RandomizedSearchCV , random_state=rn.randint(0,10)
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
print("y:",score_df["params"])     # .get("min_sample_leaf")
print("z:",score_df["mean_test_score"])
print("columns name",list(score_df.columns))
plt.show()
best_parameters[v_param_index][0] = "DecisionTreeClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion



v_param_index += 1




#region AdaBoostClassifier
modelAda = AdaBoostClassifier()   #n_estimators = 2
estimators_range = list(np.arange(1,21,2))    #list(np.arange(1,16,2))
learning_rate_range = list(np.arange(0.1,4.1,0.25))      #list(np.arange(0.1,4.1,0.8))
param_grid = dict(n_estimators=estimators_range, learning_rate=learning_rate_range)    
print(param_grid)
grid = GridSearchCV(modelAda, param_grid=param_grid, cv=skf, scoring="balanced_accuracy", verbose=3)
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
print("x:",score_df["params"])    
print("y:",score_df["params"])     
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
estimators_range = list(np.arange(1,25,2))      #list(np.arange(90,101,10)) questo è quello modificato
learning_rate_range = list(np.arange(0.1,5,0.5))      #list(np.arange(0.1,2.1,0.1)) questo è quello corretto per il plot
param_grid = dict(n_estimators=estimators_range, learning_rate=learning_rate_range)    
print(param_grid)
grid = GridSearchCV(modelGrad, param_grid=param_grid, cv=skf, scoring="balanced_accuracy", verbose=3)
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
print("x:",score_df["params"])    
print("y:",score_df["params"])     
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()
best_parameters[v_param_index][0] = "GradientBoostingClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion



v_param_index += 1



#region BaggingClassifier
modelBagging = BaggingClassifier(base_estimator=reg)
estimators_range = list(np.arange(1,30,1))
bootstrap_range = list([True,False])
param_grid = dict(n_estimators=estimators_range, bootstrap=bootstrap_range)    
print(param_grid)
grid = GridSearchCV(modelBagging, param_grid=param_grid, cv=skf, scoring="balanced_accuracy", verbose=3)
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

# converting boolean values to numerical, for visualization
for i in range(0,len(x)): 
    if x[i] == "False":
        x[i] = 0
    else:
        x[i] = 1 
    i += 1 

ax.scatter(x, y, score_df["mean_test_score"], edgecolors="red",linewidths=6)
ax.set_title('boosting classifier score')
ax.set_xlabel("bootstrap")
ax.set_ylabel("estimators number")
ax.set_zlabel("mean_test_score")
print("x:",score_df["params"])    
print("y:",score_df["params"])     
print("z:",score_df["mean_test_score"])
print("colonne",list(score_df.columns))
plt.show()
best_parameters[v_param_index][0] = "BaggingClassifier"
best_parameters[v_param_index][1] = grid.best_params_ 
best_parameters[v_param_index][2] = grid.best_score_
#endregion



'''

print("best_parameters for used models: ", best_parameters)










# Scoring overview and comparison
overall_score = [[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"]]
for element in best_parameters:
    if element[0] == "AdaBoostClassifier":
        
        ada = AdaBoostClassifier(learning_rate=element[1].get("learning_rate"))
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        print("---------------------------------------- ADABOOST CLASSIFIER ----------------------------------------")
        overall_score[0] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="AdaBoostClassifier")
    
    elif element[0] == "BaggingClassifier":
        bag = BaggingClassifier(base_estimator=reg,n_estimators=element[1].get("n_estimators"),bootstrap=element[1].get("bootstrap"))
        bag.fit(X_train, y_train)
        y_pred = bag.predict(X_test)
        print("---------------------------------------- BAGGING CLASSIFIER ----------------------------------------")
        overall_score[1] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="BaggingClassifier")
    
    elif element[0] == "GradientBoostingClassifier":
        grad = GradientBoostingClassifier(n_estimators=element[1].get("n_estimators"),learning_rate=element[1].get("learning_rate"))
        grad.fit(X_train, y_train)
        y_pred = grad.predict(X_test)
        print("---------------------------------------- GRADIENT BOOSTING CLASSIFIER ----------------------------------------")
        overall_score[2] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="GradientBoostingClassifier")
    
    elif element[0] == "DecisionTreeClassifier":
        dec = DecisionTreeClassifier(min_samples_leaf=element[1].get("min_samples_leaf"),max_depth=element[1].get("max_depth"))
        dec.fit(X_train, y_train)
        y_pred = dec.predict(X_test)
        print("---------------------------------------- DECISION TREE CLASSIFIER ----------------------------------------")
        overall_score[3] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="DecisionTreeClassifier")



reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("---------------------------------------- LOGISTIC REGRESSION ----------------------------------------")
overall_score[4] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="LogisticRegression")


dec = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1000)
bag = BaggingClassifier(base_estimator=dec,n_estimators=100)       #element[1].get("n_estimators"),bootstrap=element[1].get("bootstrap")
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
print("---------------------------------------- BAGGING CLASSIFIER ----------------------------------------")
overall_score[1] = getScoreMetrics(y_test=y_test, y_pred=y_pred, modelName="BaggingClassifier")


tempNames = [overall_score[0][4],overall_score[1][4],overall_score[2][4],overall_score[3][4],overall_score[4][4]]
print(overall_score)
for i in range(0,4):
    tempValues = [overall_score[0][i],overall_score[1][i],overall_score[2][i],overall_score[3][i], overall_score[4][i]]
    plt.bar(tempNames, tempValues)    
    if i == 0:
        plt.title("Accuracy")
    elif i == 1:
        plt.title("Recall")
    elif i == 2:
        plt.title("f1")
    elif i == 3:
        plt.title("f1 man")
        plt.show()





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

'''





 
'''
        # Plot loss (y axis) and epochs (x axis) for training set and validation set
        # plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.plot(history.epoch, np.array(history.history["loss"]), label="Train loss")
        plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Val loss")
        plt.plot(history.epoch, np.array(history.history["accuracy"]) * 100, label="Train accuracy")
        plt.plot(history.epoch, np.array(history.history["val_accuracy"]) * 100, label="Val accuracy")
        plt.legend()
    plt.show()
'''






'''

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators = 1000)
model.fit(X_train,y_train)


'''









