import json
from SMLM.generate import Generator

with open('fast.json', 'r') as f:
    config = json.load(f)

g = Generator(config)
movie, state, gtmat = g.generate()
g.save(movie,state,gtmat)
