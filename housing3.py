# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:52:41 2020

@author: DEEPAK
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

temp = pd.read_csv('train(1).csv')
df_test = pd.read_csv('test(1).csv')
df_test.set_index('Id')

temp.LotFrontage.replace(np.NaN,0,inplace=True)
df_test.LotFrontage.replace(np.NaN,0,inplace=True)
df_test['TotalBsmtSF'].replace(np.NaN,df_test.TotalBsmtSF.mean(),inplace=True)
df_test.GarageCars.replace(np.NaN,0,inplace=True)


perm = temp[['MSZoning','Neighborhood','OverallQual','BldgType','LotFrontage','GrLivArea','YearBuilt','GarageCars',
             'BedroomAbvGr','SalePrice','TotalBsmtSF']]
temp1 = perm.copy()


#plt.boxplot(perm.GrLivArea)
plt.scatter(perm.GrLivArea,perm.SalePrice)
perm = perm[perm['GrLivArea']<4000]
perm = perm[perm['TotalBsmtSF']<6000]
#perm = perm[perm['LotFrontage']<310]


perm = pd.get_dummies(perm)

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection  import train_test_split
y = perm[['SalePrice']]
X = perm.drop('SalePrice',axis=1)
X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state=42)

reg = XGBRegressor()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
y_pred=y_pred.astype(int)

cframe = pd.DataFrame({'Actual':list(y_test['SalePrice']),'Predicted':list(y_pred)},index=range(0,len(y_test)))

#y1_pred = reg.predict(df_test)
var = 'C:\\Users\\DEEPAK\\Documents\\Op1.xlsx'
id1 = df_test['Id']
df_test = df_test[['MSZoning','Neighborhood','OverallQual','BldgType','LotFrontage','GrLivArea','YearBuilt',
                   'GarageCars','BedroomAbvGr','TotalBsmtSF']]
df_test = pd.get_dummies(df_test)
pred1 = reg.predict(df_test)
pred1 = list(pred1)
pred1 = pd.DataFrame(pred1,index=id1)

with pd.ExcelWriter(var) as op:
      cframe.to_excel(op,sheet_name='Output')
var = 'C:\\Users\\DEEPAK\\Documents\\Op2.xlsx'    
with pd.ExcelWriter(var) as op:
      pred1.to_excel(op,sheet_name='Output')
      
from math import sqrt      
print(reg.score(X_test,y_test))

from sklearn.metrics import mean_squared_error
print("MSE="+str(mean_squared_error(y_test,y_pred)))
print("RMSE="+str(sqrt(mean_squared_error(y_test,y_pred))))