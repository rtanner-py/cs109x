import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('poly_2.csv')
x = df[['x']].values
y = df[['y']].values

# Plot initial data values to inspect shape
fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$')

# fit simple linear regression model
model = LinearRegression()
model.fit(x,y)
linear_predictions = model.predict(x)

# fit polynomial model. Visual inspection suggests degree=3 is suitable
x_polynomial = PolynomialFeatures(degree=3).fit_transform(x)
# Note: PolynomialFeatures adds an intercept by default so set fit_intercept to False to avoid having 2 intercepts
polynomial_model = LinearRegression(fit_intercept=False)
polynomial_model.fit(x_polynomial,y)
polynomial_predictions = polynomial_model.predict(x_polynomial)

# visualise results
x_l = np.linspace(np.min(x), np.max(x), 100).reshape(-1,1)
y_linear_predictions_l = model.predict(x_l)
x_poly_l = PolynomialFeatures(degree=3).fit_transform(x_l)
y_poly_predictions_l = polynomial_model.predict(x_poly_l)
plt.scatter(x,y,s=5,label='Data')
plt.plot(x_l, y_linear_predictions_l,label='Linear')
plt.plot(x_l, y_poly_predictions_l, label="Polynomial")
plt.legend()
plt.tight_layout()
plt.show()

# visualise residuals
poly_residuals = y - polynomial_predictions
lin_residuals = y - linear_predictions

fig, ax = plt.subplots(1,2, figsize=(10,4))
bins = np.linspace(-20,20,20)

ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')
ax[0].hist(lin_residuals, bins, label='Linear residuals')
ax[0].hist(poly_residuals, bins, label='Polynomial residuals')
ax[0].legend(loc='upper left')

ax[1].scatter(polynomial_predictions, poly_residuals, s=10)
ax[1].scatter(linear_predictions, lin_residuals, s=10)
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')

fig.suptitle('Residual analysis (Linear vs polynomial)')


