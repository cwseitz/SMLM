import json
from SMLM.generators import TimeSeries3D

with open('fast.json', 'r') as f:
    config = json.load(f)

g = TimeSeries3D(config)
g.generate()
g.save()
