import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
#generarea val pt X
m1 = stats.expon(0, 1/4).rvs(10000) 
m2 = stats.expon(0, 1/6).rvs(10000)

p1 = 0.4 #primul mecanic
p2 = 0.6 #al 2lea mecanic
x = p1 * m1 + p2 * m2

#media si deviatia
medie = np.mean(x)
deviatia = np.std(x)
print("medie:", medie)
print("deviatie", deviatia)

az.plot_posterior({'x':x})
plt.show()