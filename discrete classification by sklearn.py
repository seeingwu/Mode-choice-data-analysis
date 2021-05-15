import pandas as pd
import numpy as np
import missingno as msno
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import cross_val_score

## Make the dataset for direction 2

# load the data and check it
data = pd.read_csv('processed.csv')
data.shape
data.head()
data['Y2'].value_counts()

# select the rows that we need
df=data[(data['mode_now_Car/Van']>0)|(data['mode_now_School Bus']>0)|data['mode_now_Walk']>0]
df=df[(df['mode_before_Car/Van']>0)|(df['mode_before_School Bus']>0)|(df['mode_before_Walk']>0)]

## select the columns by deleting high-missing-value rows and delete the all NA rows.
df = df.drop(['unwill_walk'],axis=1)
df = df.dropna(axis = 0, how = 'any')

## feature selection: save the basic contributes without co-linear variables.
df = df[['age', 'IMD2019_decile', 'household_size', 'mor_travel', 'aft_travel',
       '#car',  'ethnicity_Pakistani','ethnicity_White British', 'gender_Female','Y1','Y2']]
df['Y2'].value_counts()

## have to delete the bus-walk, bus-car, walk-bus
df=df[(df['Y2']!='bus-walk')]
df=df[(df['Y2']!='bus-car')]
df=df[(df['Y2']!='walk-bus')]
df['Y2'].value_counts()


# Then use the sklearn
X = df[['age', 'IMD2019_decile', 'household_size', 'mor_travel', 'aft_travel',
       '#car',  'ethnicity_Pakistani','ethnicity_White British', 'gender_Female']]
Y = df['Y2']
X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify = Y)


# SVM via searching the best paras
# SVM with tuned parameters: kernels, C, gamma, 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

tuned_parameters = {'C': [0.1,0.01,0.001,0.0001],  
              'gamma': [0.1,0.01,0.001,0.0001], 
              'kernel': ['rbf','poly','sigmoid','linear']} 

election_model_svm = GridSearchCV(SVC(),tuned_parameters,cv=5)
ori_model_svm = election_model_svm.fit(X,Y)
print(classification_report(y_test, pred))

# fit different models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
keys = []
scores = []
models = {'K-近邻': KNeighborsClassifier(),
          '逻辑回归 l1': LogisticRegressionCV(cv=5, penalty='l1',solver='liblinear'),
          '逻辑回归 l2': LogisticRegressionCV(cv=5, penalty='l2'),
          }

for k,v in models.items():
    mod = v
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    print(str(k) + '建模效果：' + '\n')
    print(classification_report(y_test, pred))
    acc = accuracy_score(y_test, pred)
    print('分类正确率：'+ str(acc)) 
    print('\n' + '\n')
    keys.append(k)
    scores.append(acc)
    table = pd.DataFrame({'model':keys, 'accuracy score':scores})
    print(confusion_matrix(y_test, pred))

#table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


keys = []
scores = []
models = {'决策树': DecisionTreeClassifier(),
          '高斯贝叶斯': GaussianNB(),
          '伯努利贝叶斯': BernoulliNB(),
          '多项式贝叶斯': MultinomialNB()}

for k,v in models.items():
    mod = v
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    print(str(k) + '建模效果：' + '\n')
    print(classification_report(y_test, pred))
    acc = accuracy_score(y_test, pred)
    print('分类正确率：'+ str(acc)) 
    print('\n' + '\n')
    keys.append(k)
    scores.append(acc)
    table = pd.DataFrame({'model':keys, 'accuracy score':scores})
    print(confusion_matrix(y_test, pred))
