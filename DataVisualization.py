# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:40:36 2019

@author: BhamilaAuthithan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cars_data=pd.read_csv('toyota.csv',index_col=0,na_values=["??","????"])

cars_data.dropna(axis=0,inplace=True)

plt.scatter(cars_data['Age'],cars_data['Price'],c='red')

plt.title('Scatter plot of price vs age of the cars')

plt.xlabel('Age(months)')

plt.ylabel('price(Euros)')

plt.show()

plt.hist(cars_data['Km'],
         color='green',
         edgecolor='white',
         bins=5)

plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')

counts=[979,120,12]
fueltype=['Petrol','CNG','Diesel']
index=np.arange(len(fueltype))
plt.bar(index,counts,color=['red','blue','cyan'])
plt.title('Bar plot of fuel types')
plt.xlabel('Fueltypes')
plt.ylabel('Frequency')
plt.xticks(index,fueltype,rotation=90)
plt.show()





