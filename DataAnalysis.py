# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:32:34 2019

@author: BhamilaAuthithan
"""

import os
import pandas as pd

os. chdir("C:\Users\BhamilaAuthithan\Documents\Python")

cars_data = pd.read_csv('toyota.csv',index_col=0,na_values=["??","????"])

cars_data2=cars_data.copy()

print(cars_data)

pd.crosstab(index=cars_data2['Fueltype'],columns='count',dropna=True)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['Fueltype'],dropna=True)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['Fueltype'],normalize=True,dropna=True)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['Fueltype'],normalize=True,margins=True,dropna=True)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['Fueltype'],normalize='index',margins=True,dropna=True)

pd.crosstab(index=cars_data2['Automatic'],columns=cars_data2['Fueltype'],normalize='columns',margins=True,dropna=True)

numerical_data=cars_data2.select_dtypes(exclude=[object])

corr_matrix=numerical_data.corr()