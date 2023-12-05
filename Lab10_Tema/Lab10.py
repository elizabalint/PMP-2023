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

#1. 
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

# 2. 
# Calculează granița de decizie și intervalul HDI
b0_samples = trace['beta_0']
b1_samples = trace['beta_1']
b2_samples = trace['beta_2']

# Calculează probabilitatea de admitere pentru fiecare set de parametri
admission_prob_samples = 1 / (1 + np.exp(-(b0_samples + b1_samples * gre_scores + b2_samples * gpa_scores)))

# Calculează granița de decizie (unde probabilitatea este 0.5)
decision_boundary = np.median(admission_prob_samples, axis=0)

# Calculează HDI pentru granița de decizie
hdi = pm.hpd(admission_prob_samples.T, hdi_prob=0.94)

# Afișează rezultatele
plt.scatter(gre_scores, gpa_scores, c=admission, cmap='coolwarm', edgecolors='k', marker='o', label='Data')
plt.scatter(gre_scores[admission_prob_samples.mean(axis=0) >= 0.5],
            gpa_scores[admission_prob_samples.mean(axis=0) >= 0.5],
            color='red', marker='x', label='Predicted Admitted')
plt.scatter(gre_scores[admission_prob_samples.mean(axis=0) < 0.5],
            gpa_scores[admission_prob_samples.mean(axis=0) < 0.5],
            color='blue', marker='x', label='Predicted Rejected')
plt.contour(gre_scores, gpa_scores, decision_boundary.reshape(gre_scores.shape), levels=[0.5], colors='black', linewidths=2)
plt.fill_between(gre_scores, hdi[:, 0], hdi[:, 1], color='gray', alpha=0.3, label='94% HDI')
plt.xlabel('GRE Scores')
plt.ylabel('GPA Scores')
plt.title('Decision Boundary and 94% HDI')
plt.legend()
plt.show()

# 3. 
# Valorile pentru studentul specificat
new_gre_score = 550
new_gpa_score = 3.5

# Calculul probabilității pentru studentul specificat
new_admission_prob = 1 / (1 + np.exp(-(b0_samples + b1_samples * new_gre_score + b2_samples * new_gpa_score)))

# Calculul HDI pentru probabilitatea de admitere
hdi_new_admission_prob = pm.hpd(new_admission_prob, hdi_prob=0.9)

# Afișează rezultatul
print(f'Intervalul de 90% HDI pentru probabilitatea de admitere este: {hdi_new_admission_prob}')
