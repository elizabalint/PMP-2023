import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az  

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

if __name__ == '__main__':
    # Distribuția a priori pentru n (Poisson(10))
    with pm.Model() as model:
        prior_n = pm.Poisson("prior_n", mu=10)

        # Variabile pentru distribuția binomială
        binomial_distributions = []

        for Y in Y_values:
            for theta in theta_values:
                binomial_distributions.append(pm.Binomial("binomial_Y{}_theta{}".format(Y, theta), n=prior_n, p=theta, observed=Y))

        # Eșantionarea din distribuția a posteriori
        trace = pm.sample(10000, tune=1000, chains=2)

    az.plot_posterior(trace)

    plt.show()
