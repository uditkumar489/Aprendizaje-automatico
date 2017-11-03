# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:38:53 2017

NOTE : most of the pyhton libraries take care of feature scaling & linear regression is one of the case 

@author: Udit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# NO FEATURE SCALLING NEEDED


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# this code is making the machine learn or say analyse the input(X_TRAIN) vs response(Y_TRAIN) relation

# WORK_1 i.e. me ek y_test ki predicted copy (i.e. y_pred) bnana chahta hu taki me compare kar saku k predicted values of y-pred kitni close h actual response value of X_TEST (i.e. y_test)
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# WORK_2 => similarly me ab X_train k liye bhi predicted copy bna leta hu
X_pred = regressor.predict(X_train)

# WORK_3 -> aab me iss xTrain ki predicted values(i.e X_pred) ka graph bnana chahunga VS X_train ; usme original value (i.e y_train ki scattered value bhi dikha deta hu for comparison)
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, X_pred, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()