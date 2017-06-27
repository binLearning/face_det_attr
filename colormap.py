'''
Create color map for the dataset.
'''

import numpy as np


# Get the specified bit value
def _bitget(byteval, idx):
  return ((byteval & (1 << idx)) != 0)

# Create label-color map, label --- [R G B]
def labelcolormap():
  color_map = np.array([ [  0,   0,   0],
                         [  0,   0, 128],
                         [  0, 128, 128],
                         [128,   0,   0],
                         [128,   0, 128],
                         [128, 128,   0],
                         [128, 128, 128] ])
  return color_map
