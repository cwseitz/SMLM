import matplotlib.pyplot as plt
from SMLM.figures import Figure_1
import json

with open('setup3d.json', 'r') as f:
    setup3d_params = json.load(f)
    
figure = Figure_1(setup3d_params)
figure.plot()
plt.show()
