import json
from SMLM.generate import Generator2D

with open('fast.json', 'r') as f:
    config = json.load(f)

g = Generator2D(config)
movie, state, gtmat = g.generate()
mask = g.segment(gtmat,upsample=1)
g.save(movie,state,gtmat,mask)
