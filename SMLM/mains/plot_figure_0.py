import matplotlib.pyplot as plt
from SMLM.figures import Figure_0
import json

with open('setup2d.json', 'r') as f:
    setup2d_params = json.load(f)
    
figure = Figure_0(setup2d_params)
figure.plot()
plt.show()
