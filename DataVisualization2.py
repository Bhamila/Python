# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:45:18 2019

@author: BhamilaAuthithan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars_data_visual2=pd.read_csv('toyota.csv',index_col=0,na_values=['??','????'])

cars_data_visual2.dropna(axis=0,inplace=True)

sns.set(style='darkgrid')
sns.regplot(x=cars_data_visual2['Age'],y=cars_data_visual2['Price'])

sns.regplot(x=cars_data_visual2['Age'],y=cars_data_visual2['Price'],marker='*',fit_reg=False)

sns.lmplot(x='Age',y='Price',data=cars_data_visual2,fit_reg=False,hue='Fueltype',legend=True,palette="Set1")

sns.distplot(cars_data_visual2['Age'])

sns.distplot(cars_data_visual2['Age'],kde=False)

sns.distplot(cars_data_visual2['Age'],kde=False,bins=5)

sns.countplot(x="Fueltype",data=cars_data_visual2)

sns.countplot(x="Fueltype",data=cars_data_visual2,hue="Automatic")

sns.boxplot(y=cars_data_visual2["Price"])

sns.boxplot(y=cars_data_visual2["Price"],x=cars_data_visual2["Fueltype"])

sns.boxplot(y=cars_data_visual2["Price"],x=cars_data_visual2["Fueltype"],hue="Automatic",data=cars_data_visual2)

f,(ax_box,ax_hist)=plt.subplots(2,gridspec_kw={"height_ratios":(.15,.85)})

sns.boxplot(cars_data_visual2["Price"],ax=ax_box)
sns.distplot(cars_data_visual2["Price"],ax=ax_hist,kde=False)

sns.pairplot(cars_data_visual2,kind="scatter",hue="Fueltype")
plt.show()



