import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
# Subiectul 2

# definirea parametrilor distribuției normale
mu = 10  
sigma = 2  

# generare 100 de timpi medii de așteptare
timpi_medii_asteptare = np.random.normal(mu, sigma, 100)
#histograma pentru distribuția generată
plt.hist(timpi_medii_asteptare, bins=20, density=True, alpha=0.6, color='blue')

# axe
plt.title('Distribuție Normală - Timp Mediu de Așteptare')
plt.xlabel('Timp Mediu de Așteptare')
plt.ylabel('Densitate de Probabilitate')

# afisare
plt.show()

# date observate
timpi_medii_asteptare_observed = timpi_medii_asteptare


# creare model
with pm.Model() as model:
    # a priori pentru miu normala
    mu = pm.Normal('mu', mu=10, sigma=5)  

    # a priori pentru dev standard exponentiala
    sigma = pm.Exponential('sigma', lam=1) 

    # distribuție a priori pentru timpul mediu de așteptare
    timp_mediu_asteptare = pm.Normal('timp_mediu_asteptare', mu=mu, sigma=sigma, observed=timpi_medii_asteptare_observed)

    # inferenta bayesiana pentru estimarea distributiei dei probabilitate
    trace = pm.sample(1000, tune=1000)

pm.summary(trace).round(2)

# extragere lanturi de valori pentru parametrul mu din rezultatele inferentei
trace_mu = trace['mu']
# vizualizare grafica a distributiei de posteriori pentru mu
plt.hist(trace_mu, bins=30, density=True, color='blue', alpha=0.7)
plt.title('Distribuția de Posteriori pentru mu')
plt.xlabel('mu')
plt.ylabel('Densitate de Probabilitate')
plt.show()





# Subiectul 1
def aruncare_moneda(probabilitate_stema):
    return np.random.choice([0, 1], p=[1 - probabilitate_stema, probabilitate_stema])

def joc():
    # decide cine inceoe
    primul = np.random.choice([0, 1])

    # runda 1 -> primul arunca moneda
    steme_primul = aruncare_moneda(1/2)

    # runda 2 -> celalalt jucator arunca moneda de steme_primul + 1 ori
    steme_jucator_secundar = sum(aruncare_moneda(2/3) for _ in range(steme_primul + 1))

    # verificare castigator
    castigator = None
    if steme_primul >= steme_jucator_secundar:
        castigator = primul
    else:
        castigator = 1 - primul

    return castigator

# Monte Carlo
numar_jocuri = 10000
castig_j0 = 0
castig_j1 = 0

for _ in range(numar_jocuri):
    castigator = joc()
    if castigator == 0:
        castig_j0 += 1
    else:
        castig_j1 += 1

# afisare
procent_j0 = (castig_j0 / numar_jocuri) * 100
procent_j1 = (castig_j1 / numar_jocuri) * 100

print(f"J0 a castigat {procent_j0}% jocuri")
print(f"J1 a castigat {procent_j1}% jocuri")


# crearea model Bayesian
model = BayesianNetwork([('primul', 'stema1'), ('stema1', 'stema2'), ('stema1', 'castigator')])

# probabilităților conditionate 
cpd_jucator_initial = TabularCPD(variable='primul', variable_card=2, values=[[0.5, 0.5], [2/3, 1/3]],)
cpd_stema1 = TabularCPD(variable='stema1', variable_card=2, values=[[0.5, 0.5], [2/3, 1/3]],
                       evidence=['primul'], evidence_card=[2])
cpd_stema2 = TabularCPD(variable='stema2', variable_card=3, values=[[1/3, 1/3, 1/3]],
                       evidence=['stema1'], evidence_card=[2])
cpd_castigator = TabularCPD(variable='castigator', variable_card=2, values=[[1, 0, 0, 0], [0, 1, 1, 1]],
                            evidence=['stema1', 'stema2'], evidence_card=[2, 3])

# adaugarea CPD-urilor la model
model.add_cpds(cpd_jucator_initial, cpd_stema1, cpd_stema2, cpd_castigator)

# validare model
model.check_model()

# inferenta bayesiana
inferenta = VariableElimination(model)
prob_castigator = inferenta.query(variables=['castigator'], evidence={'stema1': 0, 'stema2': 0})

print(prob_castigator)
