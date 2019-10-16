# %load data_preprocessing_template.py
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# We SHOULD encode categorical varibles BEFORE splitting. 
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding DUMMY VARIABLE TRAP. Remove one of the dummy var column
X = X[:, 1:]

# NOTE: Current, python lib takes care of avoiding dummy var trap by default. SO don't need to do it manually. 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ------------- READY WITH DATA PREPROCESSING FOR THIS LEARNING----------------#

# Fitting the data into a multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Just ploting both on the same graph to compare
plt.plot(y_pred, color='blue')
plt.plot(y_test, color='red')


# USE BACKWARD ELIMINATION METHOD
import statsmodels.formula.api as sm

# append X to an array of ones
X1 = np.append(arr = np.ones((len(dataset.index), 1)).astype(int), values = X, axis= 1)

# step 1 confidence-value is 0.05

# step 2: Fit the full modell with all possible predictors/variables
X1_optimal = X1[:, [0,1,2,3,4,5]] 
# use ordinaly least significant regressor
regressor_OLS = sm.OLS(endog= y, exog= X1_optimal).fit()
regressor_OLS.summary()

# step 3, 4, 5: Find the variable with higest p-value and remove it from the model and fit again
# x2 has the highest p-value, index 2
X1_optimal = X1[:, [0,1,3,4,5]] 
# use ordinaly least significant regressor
regressor_OLS = sm.OLS(endog= y, exog= X1_optimal).fit()
regressor_OLS.summary()

# x1 has the highest p-value, index 1
X1_optimal = X1[:, [0,3,4,5]] 
# use ordinaly least significant regressor
regressor_OLS = sm.OLS(endog= y, exog= X1_optimal).fit()
regressor_OLS.summary()

# x2 has the highest p-value, index 4
X1_optimal = X1[:, [0,3,5]] 
# use ordinaly least significant regressor
regressor_OLS = sm.OLS(endog= y, exog= X1_optimal).fit()
regressor_OLS.summary()

# x2 has the highest p-value, index 4
X1_optimal = X1[:, [0,3]] 
# use ordinaly least significant regressor
regressor_OLS = sm.OLS(endog= y, exog= X1_optimal).fit()
regressor_OLS.summary()
