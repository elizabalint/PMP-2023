import pymc as pm

# 1.
if __name__=='__main__':
    model = pm.Model()

    with model:
        alpha=4
        trafic=pm.Poisson("T",20)
        plasare=pm.Normal("P", mu=2, sigma=0.5)
        pregatire=pm.Exponential("preg", 1/alpha)

   