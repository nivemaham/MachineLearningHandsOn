# %load data_preprocessing_template.py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# X = dataset.iloc[:, 1].values This makes the output of the function an array. but for ML model we want to have a vector
X = dataset.iloc[:, 1:2].values # this makes the same data into a vector instead of an array.
y = dataset.iloc[:, 2].values

# We SHOULD encode categorical varibles BEFORE splitting. 
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()

# # avoiding DUMMY VARIABLE TRAP. Remove one of the dummy var column
# X = X[:, 1:]

# # NOTE: Current, python lib takes care of avoiding dummy var trap by default. SO don't need to do it manually. 

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# We cannot split the data here, since we have very little amount of data so we will use all of the data to predict

# ------------- READY WITH DATA PREPROCESSING FOR THIS LEARNING----------------#

# STEP 1: Fitting the data into a multiple linear regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# We have TWO steps here. 1st we will create a LR, then we will add some Polynomial features to the existing regressor
# STEP 2: Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree=2)
X_polynomial = polynomialRegressor.fit_transform(X) # we want to fit the X, then transform it into a polynimial vector.

X_polynomial # it has bo +b1*x1 + b2*X1^2

# STEP 3: Fitting the newly created polynomial model into a Linear Regressor
polyLinearRegressor = LinearRegression()
polyLinearRegressor.fit(X_polynomial, y)

# STEP 4: Visualizing the Linear Regressor
plt.scatter(X,y, color='blue')
plt.plot(X, linearRegressor.predict(X), color='black')
plt.plot(X, polyLinearRegressor.predict(X_polynomial), color='Orange')

# STEP 5: Try changing the DEGREE of the polynomialFeatures
polynomialRegressor = PolynomialFeatures(degree=3)
X_polynomial3 = polynomialRegressor.fit_transform(X)
polyLinearRegressor.fit(X_polynomial3, y)
plt.plot(X, polyLinearRegressor.predict(X_polynomial3), color='pink')

polynomialRegressor = PolynomialFeatures(degree=4)
X_polynomial3 = polynomialRegressor.fit_transform(X)
polyLinearRegressor.fit(X_polynomial3, y)
plt.plot(X, polyLinearRegressor.predict(X_polynomial3), color='green')


# STEP 6: Change the resolution of X
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, polyLinearRegressor.predict(polynomialRegressor.fit_transform(X_grid)), color='red')


# ------------- PREDICT SALARY FROM THE MODEL---------------------------#

salaryLinear = linearRegressor.predict([[6.5]])
salaryPoly = polyLinearRegressor.predict(polynomialRegressor.fit_transform(np.array(6.5).reshape(-1,1)))

