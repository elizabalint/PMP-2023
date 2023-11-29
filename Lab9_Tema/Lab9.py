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
    price, processor_frequency, log_disk_size = read_data()
    
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

    # Calcularea HDI pentru beta1 și beta2
    hdi_beta1 = az.hdi(idata.posterior['beta1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(idata.posterior['beta2'], hdi_prob=0.95)

    print(f"Estimările HDI pentru beta1: {hdi_beta1}")
    print(f"Estimările HDI pentru beta2: {hdi_beta2}")

    az.plot_trace(idata, var_names=['alpha', 'beta1', 'beta2', 'sigma'])
    plt.show()
    # Simularea prețului de vânzare așteptat (miu) pentru un computer specific
    processor_frequency_new = 33
    log_disk_size_new = np.log(540)

    # Extrage eșantioane din distribuția posterioră a parametrilor
    alpha_samples = idata.posterior['alpha'].values
    beta1_samples = idata.posterior['beta1'].values
    beta2_samples = idata.posterior['beta2'].values
    sigma_samples = idata.posterior['sigma'].values
       # Simularea prețului de vânzare așteptat pentru fiecare eșantion din distribuția posterioră
    price_pred_samples = np.random.normal(
        alpha_samples + beta1_samples * processor_frequency_new + beta2_samples * log_disk_size_new,
        sigma_samples
    )

    # Construirea intervalului de 90% HDI pentru prețul de vânzare așteptat
    hdi_price_pred = az.hdi(price_pred_samples, hdi_prob=0.9)
    print(f"Intervalul de 90% HDI pentru prețul de vânzare așteptat: {hdi_price_pred}")

    # Simularea prețului de vânzare așteptat (miu) pentru un computer specific
    processor_frequency_new = 33
    log_disk_size_new = np.log(540)

    # Simularea extragerilor din distribuția predictivă posterioară
    price_pred_posterior = pm.sample_posterior_predictive(idata, samples=5000, random_seed=1)

    # Extrage extragerile simulate pentru prețul de vânzare
    price_pred_samples = price_pred_posterior['price_pred']

    # Construirea intervalului de 90% HDI pentru prețul de vânzare simulat
    hdi_price_pred = az.hdi(price_pred_samples, hdi_prob=0.9)

    print(f"Intervalul de 90% HDI pentru prețul de vânzare simulat: {hdi_price_pred}")


if __name__ == "__main__":
    np.random.seed(1)
    main()
