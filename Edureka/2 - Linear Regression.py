# implementing linear regression using python

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# reading data
data = pd.read_csv('headbrain.csv')
print(data.shape) # printing data 
data.head()

# collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# finding y = mx + c

# mean of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# total number of values
n = len(X)

# using the formula to calculate b1 and b0
numer = 0
denom = 0

for i in range(m):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# print coefficents
print(b1, b0)


# plotting value and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# calculate line values x and y

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# ploting line
plt.plot(x, y, color = '#58b970', label = 'Regression line')
plt.scatter(X, Y, c = '#ef5423', label = 'Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()


# finding the R^2
ss_t = 0
ss_r = 0

for i in range(m):
  y_pred = b0 + b1 * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r / ss_t)
print(r2)




# by using the libraries

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# cannot use rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# creating model
reg = LinearRegression()
# fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# calculating R2 score
r2_score = reg.score(X, Y)
print(r2_score)

