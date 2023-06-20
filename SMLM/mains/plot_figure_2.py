import matplotlib.pyplot as plt
from SMLM.figures import Figure_2
import json

with open('setup3d.json', 'r') as f:
    setup3d_params = json.load(f)
    
figure = Figure_2(setup3d_params)
figure.plot()
plt.show()
