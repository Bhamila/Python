# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:41:31 2019

@author: BhamilaAuthithan
"""

import pandas as pd

import numpy as np

import seaborn as sns

ppl_charm_case=pd.read_csv('People Charm case.csv')

ppl_charm_case.info()

ppl_charm_case.isnull().sum()


ppl_charm_case.lastEvaluation.quantile([0.25,0.50,0.75])

sns.boxplot(y=ppl_charm_case['numberOfProjects'])

pd.crosstab(ppl_charm_case['dept'],columns=ppl_charm_case['salary'],normalize=True)

ppl_charm_case['numberOfProjects'].median()


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

new_data=pd.get_dummies(ppl_charm_case,drop_first=True)

#Storing the column names

columns_list=list(new_data.columns)
print(columns_list)

#To separate the input names from data
features=list(set(columns_list)-set(['left']))
print(features)

#store the output names in y



y=new_data['left'].values
print(y)

#Storing the values from input features

x=new_data[features].values
print(x)


from sklearn.model_selection import train_test_split
#Splitting the data in to train and test

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=2)

#Make an instance of the model
from sklearn import linear_model

logistic=linear_model.LogisticRegression()

#Fitting the values for x and y

logistic.fit(train_x,train_y)

logistic.coef_

logistic.intercept_

#prediction from test data

prediction=logistic.predict(test_x)
print(prediction)

#Confusion matrix

confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#calculate the accuracy

accuracy_score=accuracy_score(test_y,prediction)

print(accuracy_score)

#Printing the misclassified values from prediction

print("miscalssified samples %d" %(test_y!=prediction).sum())

sns.distplot(ppl_charm_case['avgMonthlyHours'],kde=False)

from sklearn.neighbors import KNeighborsClassifier

#importing library for plotting
import matplotlib.pyplot as plt

#Storing the K nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=2)

#Fitting the values of x and y

KNN_classifier.fit(train_x,train_y)

#Predicting the test values with model
prediction=KNN_classifier.predict(test_x)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#performance metric check

confusion_matrix=confusion_matrix(test_y,prediction)

print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

#calculating the accuracy

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print("miscalssified samples %d" %(test_y!=prediction).sum())

sns.boxplot('lastEvaluation','numberOfProjects',data=ppl_charm_case)

sns.countplot(ppl_charm_case['lastEvaluation'],ppl_charm_case['numberOfProjects'])
