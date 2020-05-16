# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:07:36 2019

@author: BhamilaAuthithan
"""

import os
import pandas as pd

os. chdir("C:\\Users\\BhamilaAuthithan\\Documents")

data=pd.read_csv('microlending_data.csv')

data.info()

data.isnull().sum()

data['borrower_genders'].fillna(data['borrower_genders'].mean()) 
data['borrower_genders'].fillna(data['borrower_genders'].mean()[0]) 
data['borrower_genders'].fillna(data['borrower_genders'].median()[0])
data3=data.copy(deep=True)
data['borrower_genders'].fillna(data['borrower_genders'].mode()[0]) 
data['loan_amount'].value_counts()

import seaborn as sns
loan_amount=sns.countplot(data['loan_amount'])

print(loan_amount)

data3=data3.dropna(axis=0)

data3.replace(to_replace='2 Years',value=12,inplace=True)


data3['term_in_months']=data3['term_in_months'].astype(int)

data3.info()

cols=['activity','country_code','distribution_model']
data3=data3.drop(cols,axis=1)

new_data.info()


new_data=pd.get_dummies(data3,drop_first=True)

columns_list=list(new_data.columns)
print(columns_list)

#To separate the input names from data
features=list(set(columns_list)-set(['term_in_months']))
print(features)

#store the output names in y

y=new_data['term_in_months'].values
print(y)

#Storing the values from input features

x=new_data[features].values
print(x)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 import sklearn.linear_model as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of the model

logistic=sk.LogisticRegression()

#Fitting the values for x and y

logistic.fit(train_x,train_y)

prediction=logistic.predict(test_x)
print(prediction)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score=accuracy_score(test_y,prediction)

print(accuracy_score)

data2['term_in_months'].value_counts()
data2['borrower_genders'].value_counts()

data2.corr()

data3=pd.read_csv('lendingdata.csv')

data3.corr()
