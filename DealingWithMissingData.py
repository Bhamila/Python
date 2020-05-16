# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:58:01 2019

@author: BhamilaAuthithan
"""

import os
import pandas as pd


os.chdir("C:\Users\BhamilaAuthithan\Documents\Python")

cars_data=pd.read_csv('Toyota.csv',index_col=0,na_values=["??","????"])

cars_data2=cars_data.copy()
cars_data3=cars_data.copy()

print(cars_data2.isna().sum())

print(cars_data2.isnull().sum())

missing=cars_data2[cars_data2.isnull().any(axis=1)]

cars_data2.describe()

cars_data2['Age'].mean()

cars_data2['Age'].fillna(cars_data2['Age'].mean(),
          inplace=True)


cars_data2['KM'].mean()
cars_data2['KM'].median()

cars_data2['KM'].fillna(cars_data2['KM'].median(),
          inplace=True)

cars_data2['HP'].mean()
cars_data2['HP'].median()

cars_data2['HP'].fillna(cars_data2['HP'].mean(),
          inplace=True)

cars_data2.isnull().sum()

cars_data2['FuelType'].value_counts().index[0]

cars_data2['FuelType'].fillna(cars_data2['FuelType'].value_counts().index[0],
          inplace=True)

cars_data2['MetColor'].mode()

cars_data2['MetColor'].fillna(cars_data2['MetColor'].mode()[0],
          inplace=True)

cars_data3=cars_data3.apply(lambda x:x.fillna(x.mean())
if x.dtype=='float'
else x.fillna(x.value_counts().index[0]))

