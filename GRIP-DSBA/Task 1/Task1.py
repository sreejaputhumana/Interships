# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:34:01 2020

@author: sreeja
"""

#import libraries
import pandas as pd
import numpy as np
#read data from file
data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
#check for null values
data.isnull().sum(axis=0)
#split the data set into 2  sections

x = data.iloc[:, :-1]
y = data.iloc[:,  -1]

#create test and train data
 
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(x,y,test_size = .3,random_state =1234)

from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(x_train,y_train)

# Predict the results
y_predict = lr.predict(x_test)

# Get the R-Squared 
slr_score = lr.score(x_test, y_test)

# Coefficient and Intercept
slr_coefficient = lr.coef_
slr_intercept = lr.intercept_

#Equ of the line is y = 8.84 * x + 4.84 

#calculate the error using rmse
from sklearn.metrics import mean_squared_error
import math
lr_rmse = math.sqrt(mean_squared_error(y_test,y_predict))

import matplotlib.pyplot as plt

plt.scatter(x_test, y_test)
plt.plot(x_test,y_predict)
plt.xlabel('hours')
plt.ylabel('scores')

plt.show()




