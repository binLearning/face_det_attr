'''
Transfer weights by copying matching parameters, coercing parameters of incompatible shape,
initialize weights of Deconv layer to bilinear kernels for interpolation.
'''

from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import caffe

from surgery import transplant,interp


def main():
  rt_dir = './model'

  model_src   = os.path.join(rt_dir, sys.argv[1])
  weights_src = os.path.join(rt_dir, sys.argv[2])
  model_dst   = os.path.join(rt_dir, sys.argv[3])
  weights_dst = os.path.join(rt_dir, sys.argv[4])

  caffe.set_mode_cpu()

  net_src = caffe.Net(model_src, weights_src, caffe.TRAIN)
  net_dst = caffe.Net(model_dst, caffe.TRAIN)

  # net architecture
  print('======== source network architecture ========')
  for layer_name, blob in net_src.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))
  print('====== destination network architecture =====')
  for layer_name, blob in net_dst.blobs.iteritems():
    print(layer_name + '\t' + str(blob.data.shape))

  # net parameters
  print('========= source network parameters =========')
  for layer_name, param in  net_src.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape))# , str(param[1].data.shape)
  print('======= destination network parameters ======')
  for layer_name, param in  net_dst.params.iteritems():
    print(layer_name + '\t' + str(param[0].data.shape))# , str(param[1].data.shape)

  # transfer
  # copy parameters source net => destination net
  print('================= transfer ==================')
  transplant(net_dst, net_src)

  # initialize
  # use bilinear kernels to initialize Deconvolution layer
  print('================ initialize =================')
  interp_layers = [k for k in net_dst.params.keys() if 'up' in k]
  for k in interp_layers:
    print(k)
  interp(net_dst, interp_layers)

  print('=============================================')

  # save new weights
  net_dst.save(weights_dst)


if __name__ == '__main__':
  main()
