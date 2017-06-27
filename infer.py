import sys

import numpy as np

##from PIL import Image
from scipy.misc import imread, imsave
from skimage.color import label2rgb

import caffe

from colormap import labelcolormap


def main():
  # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
  im = imread(sys.argv[1])
  in_ = np.array(im, dtype=np.float32)
  in_ = in_[:,:,::-1]
  in_ -= np.array((104.00698793,116.66876762,122.67891434))
  in_ = in_.transpose((2,0,1))

  # load net
  net = caffe.Net('./model/face_fcn32s_deploy.prototxt',
                  './snapshot/face_fcn32s_iter_XXX.caffemodel',
                  caffe.TEST)
  # shape for input (data blob is N x C x H x W), set data
  net.blobs['data'].reshape(1, *in_.shape)
  net.blobs['data'].data[...] = in_
  # run net and take argmax for prediction
  net.forward()
  out = net.blobs['score'].data[0].argmax(0).astype(np.uint8)
  
  cmap = labelcolormap()
  
  label_mask = label2rgb(out, colors=cmap[1:], bg_label=0)
  label_mask[out == 0] = [0, 0, 0]
  imsave(sys.argv[2], label_mask.astype(np.uint8))

  
if __name__ == '__main__':
  main()
