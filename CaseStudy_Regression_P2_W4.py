# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:20:22 2019

@author: BhamilaAuthithan
"""

#Predicting price for pre owned cars

import pandas as pd
import numpy as np
import seaborn as sns

#Setting dimensions for plot

sns.set(rc={'figure.figsize':(11.7,8.27)})

#Reading the csv file

cars_data=pd.read_csv('cars_sampled.csv')

#create a copy

cars=cars_data.copy() 

#Structure of dataset

cars.info()

#Summariizing data

cars.describe()

#To set the floting precision to 3
pd.set_option('display.float_format',lambda x: '%.3f' %x)

cars.describe()

#To display maximum set of columns

pd.set_option('display.max_columns',500)

cars.describe()

#Dropping the unwanted columns

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']

cars=cars.drop(columns=col,axis=1)

#Removing duplicate records

cars.drop_duplicates(keep='first',inplace=True)
#470 duplicate records

#Data Cleaning

cars.isnull().sum()

#variable yearof registration

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()

sum(cars['yearOfRegistration']> 2018)
sum(cars['yearOfRegistration']< 1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,
            fit_reg=False,data=cars)

#working range 1950 to 2018

#variable price

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])

sum(cars['price']>1500000)
sum(cars['price']<100)

#working range 100 to 1500000

#Variable PowerPs

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#Working range 10 and 500

#Working range of data

cars=cars[
        (cars.yearOfRegistration <=2018)
        & (cars.yearOfRegistration >=1950)
        & (cars.price >=100)
        & (cars.price <=1500000)
        & (cars.powerPS >=10)
        & (cars.powerPS <=500)]

#-6700 records are dropped

#Further to simplify variable reduction

#Combining year and month of registration

cars['monthOfRegistration']/=12

#Creating new variable age by adding year and month of registration

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars.Age=cars.Age.round(2)
cars['Age'].describe()
cars.info()

# Dropping year and month of registration

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#Visualizing parameters

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#PowerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Visualizing using parmaters after narrowing working range
#Age vs Price

sns.regplot(x='Age',y='price',scatter=True,
            fit_reg=False,data=cars)

#PowerPS vs Price

sns.regplot(x='powerPS',y='price',scatter=True,
            fit_reg=False,data=cars)

#Variable seller

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)

sns.countplot(x='seller',data=cars)

#Variable offertype
cars['offerType'].value_counts()

sns.countplot(x='offerType',data=cars)

#variable abtest

cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)

sns.countplot(x='abtest',data=cars)

sns.boxplot(x='abtest',y='price',data=cars)

#Variable vehicleType

cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)

sns.countplot(x='vehicleType',data=cars)

sns.boxplot(x='vehicleType',y='price',data=cars)

#Variable gearbox

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)

sns.countplot(x='gearbox',data=cars)

sns.boxplot(x='gearbox',y='price',data=cars)

#variable model

cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)

sns.countplot(x='model',data=cars)

sns.boxplot(x='model',y='price',data=cars)

#Variable kilometer

cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)

sns.countplot(x='kilometer',data=cars)

sns.boxplot(x='kilometer',y='price',data=cars)

#Variable fuelType

cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)

sns.countplot(x='fuelType',data=cars)

sns.boxplot(x='fuelType',y='price',data=cars)

#variable brand

cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)

sns.countplot(x='brand',data=cars)

sns.boxplot(x='brand',y='price',data=cars)

#Variable notRepairedDamage

cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)

sns.countplot(x='notRepairedDamage',data=cars)

sns.boxplot(x='notRepairedDamage',y='price',data=cars)

#remove insignificant variable

col=['seller','offerType','abtest']
cars=cars.drop(col,axis=1)
cars_copy=cars.copy()

#Correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
correlation.round(1)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#We are going to build linear regression and random forest model

#1. omitting missing values
#2. include the missing values

#Omitting missing values

cars_omit=cars.dropna(axis=0)

#converting categorical variable to dummy variable

cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#Importing necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Model Building with omitted data

#separating imput and output features

x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#plotting the variable price

prices=pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

#transferring price as a logorithmic value
y1=np.log(y1)

#Splitting data in to test and train

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#baseline model for omitted data

#Finding mean for test data value

base_pred=np.mean(y_test)
print(base_pred)

#repeating same till leangth of test data

base_pred=np.repeat(base_pred,len(y_test))

#Finding RMSE(Root mean squared Error)

base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)

#Linear regression with omitted data

#Setting intercept as true

lgr=LinearRegression(fit_intercept=True)

#Model

model_lin1=lgr.fit(x_train,y_train)

#Predicting model on test set

cars_predictions_lin1=lgr.predict(x_test)

#computing MSE and RMSE 

lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)

lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R Squared value

r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnostics - residual plot analysis

residual1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residual1,scatter=True,
            fit_reg=False)
residual1.describe()

#Random forest model

#Model parameters
rf=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=100,min_samples_split=10,
                         min_samples_leaf=4,random_state=1)

#model

model_rf1=rf.fit(x_train,y_train)

#Predicting model on test set

cars_predictions_rf1=rf.predict(x_test)

#computing MSE and RMSE 

rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)

rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#R squared value

r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)

print(r2_rf_test1,r2_rf_train1)

#Model building with inputed data

cars_inputed=cars.apply(lambda x:x.fillna(x.median())
                        if x.dtype=='float' else 
                        x.fillna(x.value_counts().index[0]))

cars_inputed.isnull().sum()

#Converting categorical variables to dummy variables

cars_inputed=pd.get_dummies(cars_inputed,drop_first=True)

#Model built with inputed data

#separating imput and output features

x2=cars_inputed.drop(['price'],axis='columns',inplace=False)
y2=cars_inputed['price']

#plotting the variable price

prices=pd.DataFrame({"1. Before":y2,"2. After":np.log(y2)})
prices.hist()

#transferring price as a logorithmic value
y2=np.log(y2)

#Splitting data in to test and train

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)

#baseline model for omitted data

#Finding mean for test data value

base_pred=np.mean(y_test1)
print(base_pred)

#repeating same till leangth of test data

base_pred=np.repeat(base_pred,len(y_test1))

#Finding RMSE(Root mean squared Error)

base_root_mean_square_error=np.sqrt(mean_squared_error(y_test1,base_pred))

print(base_root_mean_square_error)

#Linear regression with inputed data

#Setting intercept as true

lgr=LinearRegression(fit_intercept=True)

#Model

model_lin2=lgr.fit(x_train1,y_train1)

#Predicting model on test set

cars_predictions_lin2=lgr.predict(x_test1)

#computing MSE and RMSE 

lin_mse2=mean_squared_error(y_test1,cars_predictions_lin2)

lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

#R Squared value

r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#Regression diagnostics - residual plot analysis

residual2=y_test1-cars_predictions_lin2
sns.regplot(x=cars_predictions_lin2,y=residual2,scatter=True,
            fit_reg=False)
residual2.describe()

#Random forest model

#Model parameters
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=100,min_samples_split=10,
                         min_samples_leaf=4,random_state=1)

#model

model_rf2=rf2.fit(x_train1,y_train1)

#Predicting model on test set

cars_predictions_rf2=rf2.predict(x_test1)

#computing MSE and RMSE 

rf_mse2=mean_squared_error(y_test1,cars_predictions_rf2)

rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R squared value

r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)

print(r2_rf_test2,r2_rf_train2)

