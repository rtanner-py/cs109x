"""
The goal of this exercise is to see how multi-collinearity can affect the predictions of a model.
For this, perform a multi-linear regression on the given dataset and compare the coefficients with those from simple linear regression of the individual predictors.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('colinearity.csv')
X = df.drop(['y'], axis=1)
y = df.y

# Look at simple linear regression model with each variable independantly
linear_coef = []
for i in X:
    x = df[[i]]
    model = LinearRegression().fit(x,y)
    linear_coef.append(model.coef_)

# multi-linear regression model
multi_LR_model = LinearRegression().fit(X,y)
multi_coef = multi_LR_model.coef_

# look at coefficients
print("Single variables:")
for i, coef in enumerate(linear_coef):
    print(f"Value of beta{i+1}: {linear_coef[i][0]:.2f}")
print('--'*10)

print('By multi-Linear regression on all variables')
for i in range(4):
    pprint(f'Value of beta{i+1} = {round(multi_coef[i],2)}')

"""
Single variables:
Value of beta1: 34.73
Value of beta2: 68.63
Value of beta3: 59.40
Value of beta4: 20.92
--------------------
By multi-Linear regression on all variables
'Value of beta1 = -24.61'
'Value of beta2 = 27.72'
'Value of beta3 = 37.67'
'Value of beta4 = 19.27'
"""

# pair-plot to see correlation
g = pd.plotting.scatter_matrix(X, figsize=(10,10), marker='o', hist_kwds = {'bins': 10}, s = 60, alpha=0.8)
plt.show()

# note correlation between x1 and x4
df['x1'].corr(df['x4'])
# np.float64(0.8000162736634897)

# correlation matrix for all variables
correlation_matrix = df[['x1','x2','x3','x4']].corr()

# heatmap for correlation matrix
fig, ax = plt.subplots()
im = ax.imshow(correlation_matrix)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))

ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j,i, f'{correlation_matrix.iloc[i,j]:.4f}',
        ha='center',va='center', color='w')

fig.tight_layout()
plt.show()

# DISCUSSION: Why do you think the coefficients change and what does it mean? 
"""
Note colinearity between x1 and x4, but correlation coefficient is 0.8.
When estimating the single variable/y relationship, the simple x1 model might be proxying for some effect of the other variables, which are directly accounted for within the multi-variable linear regression model.
"""

