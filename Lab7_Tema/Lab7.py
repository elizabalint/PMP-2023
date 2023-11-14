import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np


# a. 
data = pd.read_csv("auto-mpg.csv")

print(data.head())
plt.scatter(data['horsepower'], data['mpg'])
plt.title('Relația dintre CP și mpg')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile per galon (mpg)')
plt.show()

# b
""" # Încărcați setul de date
data = pd.read_csv('auto-mpg.csv')

# Selecția doar a coloanelor de interes
data = data[['horsepower', 'mpg']]
# Înlocuiți caracterele non-numerice cu un spațiu gol și apoi convertiți coloana la tipul numeric
data['horsepower'] = pd.to_numeric(data['horsepower'].replace('[^\d]', '', regex=True), errors='coerce')
# Înlăturați rândurile care conțin valori lipsă
data = data.dropna()

# Normalizați datele, pentru a ușura convergența modelului
data['horsepower'] = (data['horsepower'] - data['horsepower'].mean()) / data['horsepower'].std()
data['mpg'] = (data['mpg'] - data['mpg'].mean()) / data['mpg'].std()

# Definiți variabilele
X = data['horsepower'].values  # Cai putere
y = data['mpg'].values  # Mile pe galon

# Adăugați un termen de interceptare
X_centered = X - np.mean(X)

# Definiți modelul
with pm.Model() as model:
    # Coeficienții pentru regresia liniară
    alpha = pm.Normal('alpha', mu=np.mean(y), sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Modelul liniar
    mu = alpha + beta * X_centered

    # Precizia erorii
    sigma = pm.HalfNormal('sigma', sd=10)

    # Definiți distribuția așteptată a valorilor
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=y)

# Afișați modelul
pm.model_to_graphviz(model)

with model:
    # Alegem algoritmul MCMC (Metropolis)
    step = pm.Metropolis()

    # Generăm lanțurile Markov Monte Carlo
    trace = pm.sample(2000, step=step)

# Afișează rezultatele
pm.summary(trace).round(2) """



# c

# Citirea datelor din fișierul auto-mpg.csv
data = pd.read_csv('auto-mpg.csv')

# Identificarea valorilor nevalide în coloana 'horsepower'
invalid_values = data[data['horsepower'].str.contains(r'[^0-9]', na=False)]['horsepower']

# Afișarea valorilor nevalide
print("Valori nevalide în coloana 'horsepower':")
print(invalid_values)

# Înlocuirea valorilor nevalide cu NaN
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# Eliminarea rândurilor care conțin NaN
data = data.dropna(subset=['horsepower', 'mpg'])

# Calcularea mediei cailor putere și înlocuirea valorilor NaN cu media
mean_horsepower = data['horsepower'].mean()
data['horsepower'].fillna(mean_horsepower, inplace=True)

# Calcularea mediei mpg
mean_mpg = data['mpg'].mean()

# Calcularea coeficienților pentru regresia liniară
numerator = np.sum((data['horsepower'] - mean_horsepower) * (data['mpg'] - mean_mpg))
denominator = np.sum((data['horsepower'] - mean_horsepower) ** 2)

slope = numerator / denominator
intercept = mean_mpg - slope * mean_horsepower

# Afisarea rezultatelor
print("Coeficient de înclinare (slope):", slope)
print("Termen liber (intercept):", intercept)

# Construirea dreptei de regresie
regression_line = slope * data['horsepower'] + intercept

# Trasarea datelor și dreptei de regresie
plt.scatter(data['horsepower'], data['mpg'], label='Date observate')
plt.plot(data['horsepower'], regression_line, color='red', label='Dreapta de regresie')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.legend()
plt.show()
