import json
from SMLM.generators import TimeSeries2D

with open('fast.json', 'r') as f:
    config = json.load(f)

g = TimeSeries2D(config)
g.generate()
g.segment()

