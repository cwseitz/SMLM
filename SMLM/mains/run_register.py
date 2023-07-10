import tifffile
from pystackreg import StackReg
from skimage.io import imsave
import json

prefixes = [
'230707_Hela-fixed_j646_50pm overnight_20mW_2000frames_buffer-2-h2b'
]

with open('storm2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    sr = StackReg(StackReg.RIGID_BODY)
    path = config['datapath']+prefix
    stack = tifffile.imread(path+'.tif')
    out = sr.register_transform_stack(stack, reference='first')
    path = config['datapath']+prefix
    imsave('-regi.tif',out)
