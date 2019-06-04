# Multiple Linear Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#Creating dummy variables to avoid ordinal
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummt variable trao
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimaL model using Backward Elimination
# Great team of independent variable which is highly statistically significant
import statsmodels.formula.api as sm 
#adding columns of 1s
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis=1)

X_opt = X[:,[0,1,2,3,4,5]]
# regressor from ols class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Removing var with highest p values ie 2
X_opt = X[:,[0,1,3,4,5]]
# regressor from ols class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing var with highest p values ie 4
X_opt = X[:,[0,3,4,5]]
# regressor from ols class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing var with highest p values ie 5
X_opt = X[:,[0,3,5]]
# regressor from ols class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing var with highest p values ie 
X_opt = X[:,[0,3]]
# regressor from ols class
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()