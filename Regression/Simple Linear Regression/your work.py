"""
simple linear regression: 
y = b(0) + b(1)*x(1)
just like the equation for a slope y = mx + b
just keep in mind which is the dependent/independent variables
"""

#first we'll pre-process the data just like we did before

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling will be taken care of by the library we're using below

from sklearn.linear_model import LinearRegression

#create an object from the LinearRegression class
regressor = LinearRegression()

#fit and transform the data (taken care of by the LinearRegression class)
regressor.fit(X_train, y_train)

#predicting the test set results (we're creating a vector of predicted salaries)
#this is where it all comes together to predict salaries for the employees, after this is just visualizing the results to get a better grip on them
y_pred = regressor.predict([[1.5]])

#visualizing the training set using the matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='green')
plt.scatter(X_test, y_pred, color = 'red')

#plt.plot makes (at least one of it's functions) a solid line
#so it's plt.plot('x coordinates','y coordinates', other stuff)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show signifies this is the end of the graph and we want to plot/show it. It's necessary with mutiple graphs
plt.show()

# Visualising the Test set results, c is color
plt.scatter(X_test, y_test, c = 'green',marker='^')
plt.scatter(X_test,y_pred, c = 'red',marker='^')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()





