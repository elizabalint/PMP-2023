import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#prior = (grid <= 0.5).astype(int):
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)

plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

#prior = abs(grid - 0.5):
def posterior_grid_abs(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)  
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid_abs(points, h, t)

plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()


#2. 
def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error

# valorile pentru N
Ns = [100, 1000, 10000]

# pi si eroarea pentru fiecare N
pi_values = []
error_values = []

for N in Ns:
    pi, error = estimate_pi(N)
    pi_values.append(pi)
    error_values.append(error)

# media si deviatia standard a erorilor
mean_error = np.mean(error_values)
std_dev_error = np.std(error_values)

# rezultatele utilizand plt.errorbar()
plt.errorbar(Ns, pi_values, yerr=error_values, fmt='o-', capsize=5, label='Estimation')
plt.axhline(y=np.pi, color='r', linestyle='--', label='True Value of π')

plt.xscale('log')  # scala logaritmica pentru N pentru a vizualiza mai bine rezultatele
plt.xlabel('Number of Points (N)')
plt.ylabel('Estimated Value of π')
plt.title('Estimation of π with Error Bars')
plt.legend()
plt.show()

# media si deviatia standard a erorilor
print(f'Mean Error: {mean_error:.4f}%')
print(f'Standard Deviation of Error: {std_dev_error:.4f}%')


