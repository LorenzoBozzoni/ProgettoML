'''from typing import ChainMap
from matplotlib import pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(np.arange(2,8),np.arange(2,8),np.arange(2,8),edgecolors="green",linewidths=6)
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")
plt.savefig("test.pdf")
plt.show()



'''

'''

import numpy as np

parameters = """[{'max_depth': 2, 'min_samples_leaf': 1000}
 {'max_depth': 2, 'min_samples_leaf': 2000}
 {'max_depth': 2, 'min_samples_leaf': 3000}
 {'max_depth': 2, 'min_samples_leaf': 4000}
 {'max_depth': 2, 'min_samples_leaf': 5000}
 {'max_depth': 3, 'min_samples_leaf': 1000}
 {'max_depth': 3, 'min_samples_leaf': 2000}
 {'max_depth': 3, 'min_samples_leaf': 3000}
 {'max_depth': 3, 'min_samples_leaf': 4000}
 {'max_depth': 3, 'min_samples_leaf': 5000}
 {'max_depth': 4, 'min_samples_leaf': 1000}
 {'max_depth': 4, 'min_samples_leaf': 2000}
 {'max_depth': 4, 'min_samples_leaf': 3000}
 {'max_depth': 4, 'min_samples_leaf': 4000}
 {'max_depth': 4, 'min_samples_leaf': 5000}
 {'max_depth': 5, 'min_samples_leaf': 1000}
 {'max_depth': 5, 'min_samples_leaf': 2000}
 {'max_depth': 5, 'min_samples_leaf': 3000}
 {'max_depth': 5, 'min_samples_leaf': 4000}
 {'max_depth': 5, 'min_samples_leaf': 5000}
 {'max_depth': 6, 'min_samples_leaf': 1000}
 {'max_depth': 6, 'min_samples_leaf': 2000}
 {'max_depth': 6, 'min_samples_leaf': 3000}
 {'max_depth': 6, 'min_samples_leaf': 4000}
 {'max_depth': 6, 'min_samples_leaf': 5000}
 {'max_depth': 7, 'min_samples_leaf': 1000}
 {'max_depth': 7, 'min_samples_leaf': 2000}
 {'max_depth': 7, 'min_samples_leaf': 3000}
 {'max_depth': 7, 'min_samples_leaf': 4000}
 {'max_depth': 7, 'min_samples_leaf': 5000}]"""

xy_list = parameters [:-1].split("\n")

x = np.empty(len(xy_list))
y = np.empty(len(xy_list))

counter = 0
for element in xy_list:
    x[counter] = element[:-1].split(", ")[0].split(":")[1]
    y[counter] = element[:-1].split(", ")[1].split(":")[1]
    counter += 1

print(x)
print(y)

# {'max_depth': 2, 'min_samples_leaf': 1000}
#ValueError: could not convert string to float: " 1000}\n {'max_depth'"

'''

'''
import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='petal_length', size='petal_length', size_max=18,
              symbol='species', opacity=0.7)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show
'''

'''

from sklearn.ensemble import AdaBoostClassifier


v_param_index = 0
param_dict = {}




best_parameters = [[0,0],[0,0]]

param_dict["parametro11"] = "valore11"
param_dict["parametro12"] = "valore12"
best_parameters[v_param_index][0] = "modello1"
best_parameters[v_param_index][1] = param_dict
v_param_index += 1


param_dict = {}
param_dict["parametro21"] = "valore21"
param_dict["parametro22"] = "valore22"
best_parameters[v_param_index][0] = "modello2"
best_parameters[v_param_index][1] = param_dict
v_param_index += 1




print(best_parameters[0][1].get("parametro11"))


ciao = {"chiave":"valore"}





overall_score = []


for element in best_parameters:
    if element[0] == "AdaBoostClassifier":
        
        ada = AdaBoostClassifier(learning_rate=element[1].get("learning_rate"))
        ada.fit(X_test)
        y_pred = ada.predict(y_test)

        #e qui ci mettiamo una bella confusion matrix


'''

'''
import pandas as pd

column_names = ["EpochNumber", "BatchSize", "Precision","Loss"]
dfToScatter = pd.DataFrame(columns = column_names)
e_element = 10
b_element = 10
test_pr = 10
test_loss = 10
dfToAppend = pd.DataFrame([[e_element, b_element, test_pr, test_loss]]  , columns=column_names)
dfToScatter = dfToScatter.append(dfToAppend)
print(dfToAppend)
print(dfToScatter)'''

'''
import numpy as np


overall_score = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
#for i in overall_score:
    #print(i[2])
#print(overall_score)
print(overall_score[1][2])



print(list(np.arange(0.1,4.1,0.8)))'''




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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.keras import models
from tensorflow.keras import layers
# not showing warnings in terminal window
warnings.filterwarnings("ignore")


# 10 FEATURES NUMERICHE
# 22 FEATURES CATEGORICHE



def getScoreMetrics(y_test, y_pred, modelName):
    acc = accuracy_score(y_test, y_pred)
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




dfNumeric = df  # copy of the dataframe, df contains categorical values while dfNumeric has all those values converted in numeric


#region removing outvalues
for (columnName, columnData) in df.iteritems():
    if columnData.nunique() > 7 and columnName != "Status":

        max_thresold = dfNumeric[columnName].quantile(0.95)
        min_thresold = dfNumeric[columnName].quantile(0.05)
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

sm = SMOTE(random_state=rn.randint(0,10000))
X_res, y_res = sm.fit_resample(X, y)

# dividing in training and test set for using kfold cross validation in training set in order to find best parameters 

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_res,y_res, test_size=0.20)

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


skf = StratifiedKFold(n_splits=5)

reg = linear_model.LogisticRegression(solver="liblinear",class_weight='balanced')
#scores = cross_val_score(reg, X_train,y_train, cv=skf, scoring="balanced_accuracy")                           # scores è un vettore numpy quindi bisogna vedere quello
#print(scores.mean())


best_parameters = [['DecisionTreeClassifier', {'max_depth': 9, 'min_samples_leaf': 1000}, 0.7985306934647516],
 ['AdaBoostClassifier', {'learning_rate': 0.8499999999999999, 'n_estimators': 19}, 0.7789076911492878], 
 ['GradientBoostingClassifier', {'learning_rate': 1.1, 'n_estimators': 23}, 0.8755753490508325],
 ['BaggingClassifier', {'bootstrap': True, 'n_estimators': 7}, 0.7462631439123977],
 [0, 0, 0]]


'''
best_parameters = [['DecisionTreeClassifier', {'max_depth': 9, 'min_samples_leaf': 4000}, 0.7985306934647516],
 ['AdaBoostClassifier', {'learning_rate': 2, 'n_estimators': 4}, 0.7789076911492878], 
 ['GradientBoostingClassifier', {'learning_rate': 30, 'n_estimators': 13}, 0.8755753490508325],
 ['BaggingClassifier', {'bootstrap': True, 'n_estimators': 3}, 0.7462631439123977],
 [0, 0, 0]]

'''
print("best_parameters for used models: ", best_parameters)






reg.fit(X_scaled, y_train)
y_pred_logreg = reg.predict(X_test)
print("---------------------------------------- LOGISTIC REGRESSION ----------------------------------------")



# Scoring overview and comparison
overall_score = [[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"],[0,0,0,0,"null"]]
for element in best_parameters:
    if element[0] == "AdaBoostClassifier":
        
        ada = AdaBoostClassifier(learning_rate=element[1].get("learning_rate"))
        ada.fit(X_scaled, y_train)
        y_pred_ada = ada.predict(X_test)
        print("y_pred_ada", y_pred_ada)
        print("---------------------------------------- ADABOOST CLASSIFIER ----------------------------------------")
        overall_score[0] = getScoreMetrics(y_test=y_test, y_pred=y_pred_ada, modelName="AdaBoostClassifier")
    
    elif element[0] == "BaggingClassifier":
        bag = BaggingClassifier(base_estimator=reg,n_estimators=element[1].get("n_estimators"),bootstrap=element[1].get("bootstrap"))
        bag.fit(X_scaled, y_train)
        y_pred_bag = bag.predict(X_test)
        print("y_pred_bag", y_pred_bag)
        print("---------------------------------------- BAGGING CLASSIFIER ----------------------------------------")
        overall_score[1] = getScoreMetrics(y_test=y_test, y_pred=y_pred_bag, modelName="BaggingClassifier")
    
    elif element[0] == "GradientBoostingClassifier":
        grad = GradientBoostingClassifier(n_estimators=element[1].get("n_estimators"),learning_rate=element[1].get("learning_rate"))
        grad.fit(X_scaled, y_train)
        y_pred_grad = grad.predict(X_test)
        print("y_pred_grad", y_pred_grad)
        print("---------------------------------------- GRADIENT BOOSTING CLASSIFIER ----------------------------------------")
        overall_score[2] = getScoreMetrics(y_test=y_test, y_pred=y_pred_grad, modelName="GradientBoostingClassifier")
    
    elif element[0] == "DecisionTreeClassifier":
        dec = DecisionTreeClassifier(min_samples_leaf=element[1].get("min_samples_leaf"),max_depth=element[1].get("max_depth"))
        dec.fit(X_scaled, y_train)
        y_pred_dec = dec.predict(X_test)
        print("y_pred_dec", y_pred_dec)
        print("---------------------------------------- DECISION TREE CLASSIFIER ----------------------------------------")
        overall_score[3] = getScoreMetrics(y_test=y_test, y_pred=y_pred_dec, modelName="DecisionTreeClassifier")


overall_score[4] = getScoreMetrics(y_test=y_test, y_pred=y_pred_logreg, modelName="LogisticRegression")



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














