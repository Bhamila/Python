# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:14:05 2019

@author: BhamilaAuthithan
"""

# Logistic Regression

# reindexing the salary status names to 0,1

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2,drop_first=True)

#Storing the column names

columns_list=list(new_data.columns)
print(columns_list)

#To separate the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#store the output names in y

y=new_data['SalStat'].values
print(y)

#Storing the values from input features

x=new_data[features].values
print(x)

#Splitting the data in to train and test

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of the model

logistic=LogisticRegression()

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


#Logistic Regression - removing insignificant variables

#reindexing the salary status names to 0,1
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})

print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

#Storing the column names

columns_list=list(new_data.columns)
print(columns_list)

#To separate the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#store the output names in y

y=new_data['SalStat'].values
print(y)

#Storing the values from input features

x=new_data[features].values
print(x)

#Splitting the data in to train and test

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of the model

logistic=LogisticRegression()

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


#KNN Model

#importing the library of KNN

from sklearn.neighbors import KNeighborsClassifier

#importing library for plotting
import matplotlib.pyplot as plt

#Storing the K nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#Fitting the values of x and y

KNN_classifier.fit(train_x,train_y)

#Predicting the test values with model
prediction=KNN_classifier.predict(test_x)

#performance metric check

confusion_matrix=confusion_matrix(test_y,prediction)

print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

#calculating the accuracy

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print("miscalssified samples %d" %(test_y!=prediction).sum())

#Effect of K value on classifier

Misclassified_sample=[]
#calcualting error for K values between 1 and 20

for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)
