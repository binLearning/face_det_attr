import os
import sys
import numpy as np

import caffe

import score

weights = './model/face_fcn32s_trans_init.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./model/solver.prototxt')
solver.net.copy_from(weights)

# scoring
clip = 200
val = range(162771, 182638)[:clip]

for _ in range(10000):
    solver.step(1000)
    ##score.seg_tests(solver, False, val, layer='score')
    score.seg_tests(solver, 'val_score_{0}', val, layer='score')
