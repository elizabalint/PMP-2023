import arviz as az
import matplotlib.pyplot as plt
# ex 1. 
#pt model centrat
centered_eight = az.load_arviz_data("centered_eight")
num_chains_centered = centered_eight.posterior.chain.size
total_samples_centered = centered_eight.posterior.draw.size
print("Exercitiul 1")
print("Modelul Centrat:")
print(f"Numar lanturi: {num_chains_centered}")
print(f"Marimea totala a esantionului: {total_samples_centered}")

#pt model necentrat
non_centered_eight = az.load_arviz_data("non_centered_eight")
num_chains_non_centered = non_centered_eight.posterior.chain.size
total_samples_non_centered = non_centered_eight.posterior.draw.size
print("Exercitiul 1")
print("Modelul Necentrat:")
print(f"Numar lanturi pentru: {num_chains_non_centered}")
print(f"Marimea totala a esantionului: {total_samples_non_centered}")
az.plot_trace(centered_eight, var_names="theta", figsize=(10, 5))
az.plot_trace(non_centered_eight, var_names="theta",figsize=(10, 5))
plt.show()

# ex 2. 
# rezumatul statistic pt parametrii mu si tau
centered_summary = az.summary(centered_eight, var_names=["mu", "tau"], round_to=2)
non_centered_summary = az.summary(non_centered_eight, var_names=["mu", "tau"], round_to=2)
print("Exercitiul 2")
print("Rezumat statistic pentru codelul Centrat:")
print(centered_summary)
print("\nRezumat statistic pentru modelul necentrat:")
print(non_centered_summary)
# distributiile parametrilor mu și tau 
az.plot_forest([centered_eight, non_centered_eight], model_names=["Centrat", "Necentrat"], var_names=["mu", "tau"], figsize=(12, 6))


# ex 3. 
print("Exercitiul 3")
# Afișează numărul de divergențe pentru fiecare model
print("Numărul de divergențe pentru Modelul Centrat:", centered_eight.sample_stats.diverging.sum())
print("Numărul de divergențe pentru Modelul Necentrat:", non_centered_eight.sample_stats.diverging.sum())

# Vizualizează zonele în care apar divergențe în spațiul parametrilor (mu și tau)
az.plot_pair(centered_eight, var_names=["mu", "tau"], divergences=True, figsize=(12, 6))
az.plot_pair(non_centered_eight, var_names=["mu", "tau"], divergences=True, figsize=(12, 6))
plt.show()