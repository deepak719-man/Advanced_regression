# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:25:39 2020

@author: DEEPAK
"""
import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\DEEPAK\\Downloads\\diabetes.csv")
col = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']
df[col] = df[col].replace(0,np.NaN)



df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

#print(df.isnull().sum())

X = df.drop('Outcome',axis=1)
y = df.Outcome

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

from sklearn.tree import DecisionTreeClassifier
classi = DecisionTreeClassifier()
classi.fit(X_train, y_train)

print(classi.score(X_test, y_test))