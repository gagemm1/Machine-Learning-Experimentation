# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:00:32 2019

@author: gagmorri
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable(s)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#3 again for the onehotencoder below since that's the index of the column you're changing
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap (line below eliminates column 0)
#using some libraries, this is taken care of for you
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results (we're creating a vector of predicted salaries)
#this is where it all comes together to predict salaries for the employees, after this is just visualizing the results to get a better grip on them
y_pred = regressor.predict(X_test)

#we won't be plotting the stuff since we'd need extra dimensions for the graph to get the extra independent values

#build the optimal model using backwards eliminations
import statsmodels.formula.api as sm

#we're about to add an extra column to the dataset because we're adding the b in y=mx + b for some reason
#axis 1 or 0: 1 specifies adding on the vertical axis, 0 on the horizontal axis
#we're adding the column of 1's because that adds a y-intercept (so when x = 0)
#this must be because each variable is being tested separately and therefore we need an intercept for each
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#this will in the end be the optimal matrix of independent variables
#all predictors/independent variables: X_opt = X[:, [0,1,2,3,4,5]]
#we just eliminated those with a higher p value than the 5% limit
#remember that column 0 is your intercept/ b in y=mx+b, don't mess with it
X_opt = X[:, [0,3]]

#class is to calculate ordinary least squares (having to do with error of the line you're fitting to the data, aka how close the line comes to the data)

"""
class statsmodels.regression.linear_model.OLS(endog, exog=None, missing='none', hasconst=None, **kwargs)[source]
A simple ordinary least squares model.

Parameters:	
endog (array-like) – 1-d endogenous response variable. The dependent variable.
exog (array-like) – A nobs x k array where nobs is the number of observations and k is 
the number of regressors. An intercept is not included by default and should be added by the user. See statsmodels.tools.add_constant.
missing (str) – Available options are ‘none’, ‘drop’, and ‘raise’. If ‘none’, no nan checking is done. 
If ‘drop’, any observations with nans are dropped. If ‘raise’, an error is raised. Default is ‘none.’
hasconst (None or bool) – Indicates whether the RHS includes a user-supplied constant. If True, a constant is 
not checked for and k_constant is set to 1 and all result statistics are calculated as if a constant is present. 
If False, a constant is not checked for and k_constant is set to 0.
"""
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#we've learned then that R&D spending is the strongest predictor and only one that was below the 5% bound

"""
Automatic backwards elimination with p-values only:
    
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

Automatic backwards elimination with p-values and adjusted R-squared:
    
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""