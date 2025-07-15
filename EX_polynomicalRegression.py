import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def get_poly_pred(x_train, x_test, y_train, degree=1):
    x_poly_train = PolynomialFeatures(degree=degree).fit_transform(x_train)
    print(x_train.shape, x_test.shape, y_train.shape)
    x_poly_test = PolynomialFeatures(degree=degree).fit_transform(x_test)
    polynomial_model = LinearRegression()
    polynomial_model.fit(x_poly_train, y_train)
    y_polynomial_predictions = polynomial_model.predict(x_poly_test)
    return y_polynomial_predictions

df = pd.read_csv('poly.csv')
x = df[['x']].values
y = df['y'].values

#plotting raw data to get an idea of its shape
plt.scatter(x,y, marker='x')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()
# indicates that degrees=3 would be a good fit

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=22)

#simple linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
linear_predictions = model.predict(x_test)

#plot with linear predictions
plt.scatter(x,y, marker='x')
plt.xlabel('x values')
plt.ylabel('y values')
plt.plot(x_test, linear_predictions, label="Linear predictions")
plt.legend()
plt.show()

#polynomical with degree=3
polynomial_predictions = get_poly_pred(x_train, x_test, y_train, degree=3)

# Sort data for for plotting
index = np.argsort(x_test[:,0])
x_test = x_test[index]
y_test = y_test[index]
linear_predictions = linear_predictions[index]
polynomial_predictions = polynomial_predictions[index]

plt.scatter(x,y,s=10, label="Test data")
plt.plot(x_test, linear_predictions, label="Linear fit", color='k')
plt.plot(x_test, polynomial_predictions, label='Polynomical fit', color='red', alpha=0.6)
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.show()

# look at residuals
polynomial_residuals = y_test - polynomial_predictions
linear_residuals = y_test - linear_predictions

# plot residuals
fig, ax = plt.subplots(1,2, figsize=(10,4))
bins = np.linspace(-20,20,20)

ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')
ax[0].hist(polynomial_residuals, bins,
           label='Polynomial residuals', 
           color='#B2D7D0',
           alpha=0.6)
ax[0].hist(linear_residuals, bins,
           label='Polynomial residuals', 
           color='#EFAEA4',
           alpha=0.6)
ax[0].legend(loc='upper left')
ax[1].hlines(0,-75,75, color='k', ls='--', alpha=0.3, label='Zero residual')
ax[1].scatter(polynomial_predictions, polynomial_residuals, s=10, color='#B2D7D0', label='Polynomial predictions')
ax[1].scatter(linear_predictions, linear_residuals, s = 10, color='#EFAEA4', label='Linear predictions' )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')
ax[1].legend(loc = 'upper left')
fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show()

"""
⏸ What is it about the plots above that are sign that a linear model is not appropriate for the data?

A. Residuals not normally distributed
B. Residuals distribution not clearly centered on zero
C. Residuals do not have constant variance
D. All of the above
"""
# Answer is D
