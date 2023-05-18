import json
from SMLM.generate import Generator

with open('slow.json', 'r') as f:
    config = json.load(f)

g = Generator(config)
movie, state = g.generate()
g.save(movie,state)
