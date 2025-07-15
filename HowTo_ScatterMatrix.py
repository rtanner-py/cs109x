# Boiler plate code for a scatter matrix / pair-plot
import pandas as pd
import matplotlib.pyplot as plt

# read in data
df = pd.read_csv('colinearity.csv')

# separate out dependant and independant variables
X = df.drop(['y'], axis=1)
y = df.y

# pair-plot
g = pd.plotting.scatter_matrix(X, figsize=(10,10), marker='o', hist_kwds = {'bins': 10}, s = 60, alpha=0.8)
plt.show()

# correlation matrix
correlation_matrix = df[['x1','x2','x3','x4']].corr()
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
