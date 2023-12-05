
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Citeste datele din CSV
data = pd.read_csv('Admission.csv')

# Extrage scorul GRE, GPA și admiterea
admission = data['Admission'].values
gre_scores = data['GRE'].values
gpa_scores = data['GPA'].values

# Definiți modelul în PyMC
with pm.Model() as admission_model:
    # Distribuții a priori pentru parametrii
    beta_0 = pm.Normal('beta_0', mu=0, tau=0.001)
    beta_1 = pm.Normal('beta_1', mu=0, tau=0.001)
    beta_2 = pm.Normal('beta_2', mu=0, tau=0.001)

    # Modelul logistic
    admission_prob = pm.invlogit(beta_0 + beta_1 * gre_scores + beta_2 * gpa_scores)

    # Distribuție a priori pentru admitere
    admission_likelihood = pm.Bernoulli('admission_likelihood', p=admission_prob, observed=admission)

# Simulați distribuția a posteriori folosind metoda Metropolis
with admission_model:
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step, tune=1000, chains=1)

# Analiza rezultatelor
pm.traceplot(trace)