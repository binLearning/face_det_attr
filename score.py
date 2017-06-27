from __future__ import division

import os
import sys

import numpy as np

from datetime import datetime
##from PIL import Image
from scipy.misc import imread, imsave
from skimage.color import label2rgb

import caffe

from colormap import labelcolormap


def fast_hist(a, b, n):
  k = (a >= 0) & (a < n)
  return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
  cmap = labelcolormap()

  n_cl = net.blobs[layer].channels
  if save_dir:
      os.mkdir(save_dir)
  hist = np.zeros((n_cl, n_cl))
  loss = 0
  for idx in dataset:
    net.forward()
    hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                      net.blobs[layer].data[0].argmax(0).flatten(),
                      n_cl)

    if save_dir:
      ##im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
      ##im.save(os.path.join(save_dir, idx + '.png'))
      label_predict = net.blobs[layer].data[0].argmax(0).astype(np.uint8)
      label_mask = label2rgb(label_predict, colors=cmap[1:], bg_label=0)
      label_mask[label_predict == 0] = [0, 0, 0]
      #imsave(os.path.join(save_dir, idx + '.png'), label_mask.astype(np.uint8))
      imsave(os.path.join(save_dir, '{:0>6}.jpg'.format(idx)),
             label_mask.astype(np.uint8))
    
    # compute the loss as well
    loss += net.blobs['loss'].data.flat[0]
  return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
  print '>>>', datetime.now(), 'Begin seg tests'
  solver.test_nets[0].share_with(solver.net)
  do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
  n_cl = net.blobs[layer].channels
  if save_format:
    save_format = save_format.format(iter)
  hist, loss = compute_hist(net, save_format, dataset, layer, gt)
  # mean loss
  print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
  # overall accuracy
  acc = np.diag(hist).sum() / hist.sum()
  print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
  # per-class accuracy
  acc = np.diag(hist) / hist.sum(1)
  print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
  # per-class IU
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
  freq = hist.sum(1) / hist.sum()
  print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
        (freq[freq > 0] * iu[freq > 0]).sum()
  return hist
