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
print(dfToScatter)