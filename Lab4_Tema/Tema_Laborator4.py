import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# 1. Definiți modelul pentru generarea traficului și timpul de pregătire
if __name__ == '__main__':
    model = pm.Model()

    with pm.Model():
        trafic = pm.Poisson("T", mu=20)
        plasare = pm.Normal("P", mu=2, sigma=0.5)
        pregatire = pm.Exponential("preg", 1/20)

# 2. Simulați numărul de clienți care intră într-o oră și găsiți α maxim
lambdaa = 20  # Numărul mediu de clienți pe oră

with pm.Model():
    num_clients = pm.Poisson('num_clients', lambdaa)

    def total_service_time(alpha):
        order_time = pm.Normal('order_time', mu=2, sigma=0.5, shape=num_clients)
        cook_time = pm.Exponential('cook_time', lam=1/alpha, shape=num_clients)
        total_time = pm.Deterministic('total_time', order_time + cook_time)
        return total_time

    def probability_serve_within_15_minutes(alpha):
        total_time = total_service_time(alpha)
        return pm.math.mean(total_time <= 15)

    alpha = pm.Uniform('alpha', 0, 15)
    prob_within_15_minutes = pm.Deterministic('prob_within_15_minutes', probability_serve_within_15_minutes(alpha))

    trace = pm.sample(2000, tune=1000, cores=1)

# Afișăm valoarea maximă a lui alfa care îndeplinește criteriul
alpha_max = trace['alpha'][trace['prob_within_15_minutes'].argmax()]
print(f'Alpha maxim pentru a servi clienții în mai puțin de 15 minute cu probabilitate de 95%: {alpha_max:.2f} minute')


# 3
alpha = alpha_max  # Utilizați valoarea maximă a lui alpha
total_time = total_service_time(alpha)

# Calculul timpului mediu de așteptare pentru un client
average_wait_time = total_time.mean()

print(f'Timpul mediu de așteptare al unui client: {average_wait_time:.2f} minute')
