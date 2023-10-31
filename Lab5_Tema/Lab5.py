import csv
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Listă pentru a stoca valorile de trafic
minut = []
nr_masini = []
# Citirea datelor din fișierul CSV (presupunând că aveți un fișier trafic.csv)
with open('trafic.csv', 'r') as csvfile:
    lines = csv.reader(csvfile)
    next(lines)  # Omiterea primei linii cu antetul
    for row in lines:
        minut.append(int(row[0]))
        nr_masini.append(int(row[1]))

# Adăugați zerouri la sfârșitul datelor observate pentru a ajunge la dimensiunea dorită
traffic_data = np.pad(nr_masini, (0, 1440 - len(nr_masini)), 'constant')

# Definirea modelului probabilistic
model = pm.Model()
with model:
    # Parametrul necunoscut λ pentru distribuția Poisson
    lambda_ = pm.Exponential('lambda_', lam=1.0)
    
    # Definirea momentelor de creștere și descreștere ale mediei traficului
    change_points = np.array([7, 8, 16, 19])
    
    # Distribuții pentru creștere și descreștere
    delta = pm.Normal('delta', mu=0, tau=1, shape=change_points.shape)
    
    # Definirea momentelor de timp într-o zi
    minutes_in_a_day = 60 * 24
    minute_of_day = np.arange(minutes_in_a_day)
    
    # Calculul mediei traficului pentru fiecare minut
    delta_sum = []
    for minute in minute_of_day:
        delta_sum.append(pm.math.sum(delta * pm.math.switch(change_points <= minute, 1, 0)))
    delta_sum = pm.math.stack(delta_sum)

    # Distribuția Poisson pentru datele observate
    traffic_observed = pm.Poisson('traffic_observed', mu=delta_sum, observed=traffic_data)

# Setați punctul de plecare explicit
start = {'lambda_': lambda_, 'delta': delta}

# Inferența Bayesiană
with model:
    trace = pm.sample(2000, tune=2000, cores=2)

# Afișarea rezultatelor
pm.Matplot.plot(trace)
plt.show()
