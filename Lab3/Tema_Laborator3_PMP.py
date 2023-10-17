from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Definirea modelului probabilistic
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarmă'), ('Incendiu', 'Alarmă')])

# Definirea individuală a CPD-urilor
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])  # Probabilitatea cutremurului
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, 
                          values=[[0.99, 0.97], [0.01, 0.03]],
                          evidence=['Cutremur'], evidence_card=[2])
cpd_alarmă = TabularCPD(variable='Alarmă', variable_card=2, values=[[0.9998, 0.02, 0.01, 0.98],
                                                                  [0.0002, 0.98, 0.99, 0.02]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])  # Probabilitatea alarmei dat cutremurul și incendiul

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

# Verificarea modelului
assert model.check_model()

# Crearea unui obiect de inferență
infer = VariableElimination(model)

# 1. Probabilitatea ca un incendiu să aibă loc, fără ca alarma de incendiu să se activeze
result_1 = infer.query(variables=['Incendiu'], evidence={'Alarmă': 0})
print(result_1)

# 2. Probabilitatea ca un cutremur să aibă loc, dat fiind că alarma de incendiu s-a declanșat
result_2 = infer.query(variables=['Cutremur'], evidence={'Alarmă': 1})
print(result_2)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()