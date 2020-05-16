# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:35:28 2019

@author: BhamilaAuthithan
"""

# ================================
"""Classifyting persnal income"""
#================================

import os

#to work with Dataframes
import pandas as pd

#To Visualize the data
import seaborn as sns
import matplotlib.pyplot as plt

#To perform numerical operations
import numpy as np

#To partition the data
from sklearn.model_selection import train_test_split

#Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

#Importing performance metrics - accuracy score & confusion
from sklearn.metrics import accuracy_score,confusion_matrix

#Importing data
data_income=pd.read_csv('income.csv')

#Creating a copy of original data
data=data_income.copy()

"""
Exploratory data analysis:
#1. Getting to know the data
#2. Data reprocessing (missing values)
#3. Cross tables and data visualization
"""

# To check whether there are missing values

data.isnull().sum()
print("Data columns with null values : \n",data.isnull().sum())
# Getting to know the data
# To check variables data type
print(data.info())

#Summary of numerical variables
summary_num=data.describe()
print(summary_num)

# Summary of Categorical variables
summary_cate=data.describe(include='O')
print(summary_cate)

# To check the frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

# Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#Go back and read the data by including na_vlues[" ?"] to consider as Nan
data=pd.read_csv('income.csv',na_values=[" ?"])

#Data processing

data.isnull().sum()

#To subset the data - axis=1 => to consider at least one column value is missing
missing=data[data.isnull().any(axis=1)]

# points to note
"""
1.Missing values in job type =1809
2. Missing values in occupation= 1816
3. There are 1809 where 2 specific colomns
i.e., occupation and job type have missing values
4. (1816-1809)=7 -> you still have occupation unfilled for these 7 rows 
because, job type is "never worked"
"""

#remove the missing values
data2=data.dropna(axis=0)

#To check the relationship between independent variables

correlation=data.corr()

#Cross tables & data visulaization

#extracting the column names 
data2.columns

#gender proportion table

gender=pd.crosstab(index=data2['gender'],
                   columns='count',
                   normalize=True)
print(gender)

#gender vs salary status

gender_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index') # to include the row proportion =1
print(gender_salstat)

#frequency distribution of salstat

SalStat=sns.countplot(data2['SalStat'])

""""
1. 75% of peple's salary status is <=50000
2. 25% of people's salary is >50000
""""

#Histogram of age

sns.distplot(data2['age'],bins=10,kde=False)

""""
people with age 20 - 45 are high in frequency
""""

# Box plot - Age vs Salary satus

sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

##People with 35-50 are more likely to earn > 50000 USD
##People with 25-35  are more likely to earn < 50000

#Plots

capitalloss=sns.countplot(data2['capitalloss'])


