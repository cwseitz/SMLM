import json
from SMLM.generators import TimeSeries2D

with open('fast.json', 'r') as f:
    config = json.load(f)

g = TimeSeries2D(config)
movie, state, gtmat = g.generate()
mask = g.segment(gtmat,upsample=1)
g.save(movie,state,gtmat,mask)
