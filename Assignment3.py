# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:37:03 2019

@author: BhamilaAuthithan
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("C:\Users\BhamilaAuthithan\Documents\Python")
mtcars=pd.read_csv('mtcars.csv',index_col=0)

mtcars.isnull().sum()

plt.hist(mtcars['mpg'],
         color='green',
         edgecolor='white'
         )

plt.scatter(mtcars['mpg'],mtcars['wt'],c='red')



sns.boxplot(x=diamond['price'],y=diamond['cut'])

diamond['cut'].value_counts()

churn=pd.read_csv('churn.csv',index_col=0)

churn['customerID'].value_counts()

churn['TotalCharges'].isnull().sum()

churn['MonthlyCharges'].mean()

churn['Dependents'].value_counts()

churn.tenure.dtypes