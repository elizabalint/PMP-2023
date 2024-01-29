import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
""" # a. 
df = pd.read_csv('BostonHousing.csv')

# b. 
X = df[['rm', 'crim', 'indus']].values  # variabile independente
y = df['medv'].values  # variabila dependentă

# adaugati o coloana de interceptie la matricea de variabile independente
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

# definire model
with pm.Model() as model:
    # priori pentru coeficientii
    beta_rm = pm.Normal('beta_rm', mu=0, sigma=1)
    beta_crim = pm.Normal('beta_crim', mu=0, sigma=1)
    beta_indus = pm.Normal('beta_indus', mu=0, sigma=1)
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    
    # modelul liniar
    mu = alpha + beta_rm * df['rm'] + beta_crim * df['crim'] + beta_indus * df['indus']
    
    # likelihood
    medv = pm.Normal('medv', mu=mu, sigma=1, observed=df['medv'])

# afisare 
with model:
    trace = pm.sample(1000, tune=1000)
az.plot_posterior(trace, var_names=['beta', 'sigma'], kind='hist')
az.plot_forest(trace, var_names=['beta'], combined=True)
az.plot_pair(trace, var_names=['beta'])

# c. 
az.plot_hdi(trace, hdi_prob=0.95, var_names=['beta'])
az.plot_forest(trace, var_names=['beta'], combined=True)

# d. 
with model:
    post_pred = pm.sample_posterior_predictive(trace, 1000)

medv_pred = post_pred['medv']
hdi_50 = pm.stats.hpd(medv_pred, hdi_prob=0.5)
print("Interval de predictie de 50% HDI pentru valoarea locuintelor:", hdi_50) """



# 2. 
# a. 
def posterior_grid(grid_points=50, head=4, tails=5):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = (1 - grid) ** (head - 1) * grid
    
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (4, 5))
points = 20
h = data.sum() 
t = len(data) - h

grid, posterior = posterior_grid(points, h, t)
# afisare
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# b. 
estimat = grid[np.argmax(posterior)]
print(f"Estimarea pentru θ care maximizează probabilitatea a posteriori: {estimat}")