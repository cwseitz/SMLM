import matplotlib.pyplot as plt
from SMLM.figures import Figure_1
import json

with open('setup2d.json', 'r') as f:
    setup2d_params = json.load(f)
with open('setup3d.json', 'r') as f:
    setup3d_params = json.load(f)
    
figure = Figure_1(setup2d_params,setup3d_params)
figure.plot()
plt.show()
