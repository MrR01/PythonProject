import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from collections import Counter

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.plotly as py
import plotly.graph_objs as go



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

#Load Data
os.chdir("C:/Raj/Education/DataScience/Tutorial/PythonSpyder/Adult") 

df1a = pd.read_csv('adult.txt')
df1a.shape
df1a.head(10)
df1a.describe()

#Add column headers as they were missing
df1b = pd.read_csv('adult.txt', names=['Age','WorkClass','fnlwgt','Education','Education-Num','Marital-Status', 'Occupation','Relationship','Race','Gender','Capital-Gain','Capital-Loss','Hrs-per-wk','Country','Income'], header=0)
print (df1b)

export_csv = df1b.to_csv (r'C:\Raj\Education\DataScience\Tutorial\PythonSpyder\Adult\export_dataframe.csv', index = None, header=True)

#check of any missing values
df1b = df1b[df1b.isnull().any(axis=1)]
#or
print (df1b.isnull())
#or
print (df1b.isnull().any(axis=1))
#or
df1b = df1b[df1b.isnull().any(axis=1)]
print (df1b)
#Counting cells with missing values:
sum(df1b.isnull().values.ravel())
#or
df1b.apply(lambda x: sum(x.isnull().values), axis = 1) # For rows 
#or
#Number of rows with at least one missing value:
sum(df1b.apply(lambda x: sum(x.isnull().values), axis = 1)>0)

#Feature Engineering - 
for x in df1b.columns:
    if df1b[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df1b[x].values))
        df1b[x] = lbl.transform(list(df1b[x].values))
        
 
# Explore Sex vs Income
g = sns.barplot(x="Gender",y="Income",data=df1b)
g = g.set_ylabel("Income >50K Probability")
sns.plt.show() 

trace0 = go.Scatter(
    x=["Gender"],
    y=["Income"],
    mode='markers',
    marker=dict(
        size=[40, 60, 80, 100],
    )
)

# Explore Relationship vs Income
g = sns.factorplot(x="Relationship",y="Income",data=df1b,kind="bar", size = 6,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")
sns.plt.show()

# Explore Marital Status vs Income
g = sns.factorplot(x="Marital-Status",y="Income",data=df1b,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")
sns.plt.show()

# Explore Workclass vs Income
g = sns.factorplot(x="WorkClass",y="Income",data=df1b,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Income >50K Probability")
sns.plt.show()

 
#drop columns that are irrelvant
df1c = df1b.drop(['WorkClass', 'Education' , 'Occupation', 'Relationship', 'fnlwgt', 'Capital-Loss', 'Hrs-per-wk', 'Country', 'Marital-Status'], axis=1)     

df1c.head()
df1c.dtypes

#5. Modeling
## Split-out Validation Dataset and Create Test Variables
X = df1c.iloc[:, :-1] #means taking all the values
print(X)
Y = df1c.iloc[:, 5]
print(Y)

validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)


# Params for Random Forest
num_trees = 100
max_features = 3


#Spot Check 5 Algorithms (LR, LDA, KNN, CART, GNB, SVM)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('NN', MLPClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
fig = plt.figure()
fig.suptitle('Algorith Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
   
#6. Algorithm Tuning
best_n_estimator = 250
best_max_feature = 5
# Tune Random Forest
n_estimators = np.array([50,100,150,200,250])
max_features = np.array([1,2,3,4,5])
param_grid = dict(n_estimators=n_estimators,max_features=max_features)
model = RandomForestClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) 
    
#7. Finalize Model
# 5. Finalize Model
# a) Predictions on validation dataset - KNN
random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
random_forest.fit(X_train, Y_train)
predictions = random_forest.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
    
        


