import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def read_data():
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)

    # Extrage coloanele relevante
    price = df['Price'].values.astype(float)
    processor_frequency = df['Speed'].values.astype(float)
    log_disk_size = np.log(df['HardDrive'].values.astype(float))

    return np.array(processor_frequency), np.array(log_disk_size), np.array(price)

def plot_data(processor_frequency, log_disk_size, price):
    plt.scatter(processor_frequency, price, marker='o')
    plt.xlabel('Processor Frequency (MHz)')
    plt.ylabel('Price')
    plt.title('PC Prices')
    plt.show()

def main():
    processor_frequency, log_disk_size, price = read_data()
    
    # Vizualizează datele folosind funcția definită anterior
    plot_data(processor_frequency, log_disk_size, price)

    with pm.Model() as model_regression:
        # Definirea parametrilor modelului
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        sigma = pm.HalfCauchy('sigma', 5)

        # Definirea variabilelor deterministe și a variabilei observate
        mu = pm.Deterministic('mu', alpha + beta1 * processor_frequency + beta2 * log_disk_size)
        price_pred = pm.Normal('price_pred', mu=mu, sigma=sigma, observed=price)

        # Eșantionarea din distribuția posterioră folosind MCMC
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alpha', 'beta1', 'beta2', 'sigma'])
    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    main()
