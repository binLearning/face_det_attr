from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import numpy as np

from scipy.io import savemat
from scipy.misc import imread, imsave, imresize
from shutil import rmtree
from six.moves import xrange


# calculate face attribute label
# 0 0 0 0 NO_BEARD EYEGLASSES MALE 1
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# [0, 0, 0, 1, 0, 0, 0, 2, 0, 3,  0,  4,  0,  5,  0,  6]
def calc_attr_label(attr_info):
  list_label_conversion = [0, 0, 0, 1, 0, 0, 0, 2, \
                           0, 3, 0, 4, 0, 5, 0, 6]
  
  EYEGLASSES = 15
  MALE       = 20
  NO_BEARD   = 24
  
  is_male       = attr_info[MALE] + 1
  is_eyeglasses = attr_info[EYEGLASSES] + 1
  is_no_beard   = attr_info[NO_BEARD] + 1
  
  label = is_no_beard*4 + is_eyeglasses*2 + is_male + 1
  
  # 00000X01[1,5](female have beard) should not existent
  if label == 1:
    label = 9
  if label == 5:
    label = 13
  
  return list_label_conversion[label]


# calculate facial region bounding box
# upper left, lower right
def calc_bbox(facial_point_info):
  mask_coord_x = np.array([0,1,0,1,0,1,0,1,0,1])
  mask_coord_y = np.array([1,0,1,0,1,0,1,0,1,0])
  
  array_coord_x = np.ma.array(facial_point_info, mask=mask_coord_x)
  array_coord_y = np.ma.array(facial_point_info, mask=mask_coord_y)
  
  min_x = array_coord_x.min()
  max_x = array_coord_x.max()
  min_y = array_coord_y.min()
  max_y = array_coord_y.max()
  
  w = max_x - min_x
  h = max_y - min_y
  
  crop_ul_x = min_x - w*3//5
  crop_ul_y = min_y - h*4//5
  crop_lr_x = max_x + w*3//5
  crop_lr_y = max_y + h*4//5
  
  return np.array([crop_ul_x, crop_ul_y, crop_lr_x, crop_lr_y])


def main():
  dir_src = sys.argv[1]
  dir_dst = sys.argv[2]
  
  if os.path.exists(dir_dst):
    rmtree(dir_dst)
  os.mkdir(dir_dst)
  
  attr_info = np.loadtxt('./data/list_attr_celeba_simplified.txt', dtype='int')
  facial_point_info = np.loadtxt('./data/list_landmarks_celeba_simplified.txt',
                                 dtype='int')
  
  total_num_image = attr_info.shape[0]
  total_num_attr  = attr_info.shape[1]
  print(total_num_image, total_num_attr)
  
  image_name_form = '{:0>6}.jpg'
  mat_name_form = '{:0>6}.mat'
  
  #Check failed: error == cudaSuccess (2 vs. 0)  out of memory
  #MAX_HW = 2890000 # 1700*1700
  #When training + test, Check failed: error == cudaSuccess (2 vs. 0)  out of memory
  #MAX_HW = 2250000 # 1500*1500
  
  MAX_HW = 1000000 # 1000*1000
  
  for proc_num in xrange(total_num_image):
    image_name = image_name_form.format(proc_num+1)
    mat_name = mat_name_form.format(proc_num+1)
    
    label = calc_attr_label(attr_info[proc_num])
    #print(image_name, label)
    
    image_data = imread(os.path.join(dir_src,image_name))
    bbox_coord = calc_bbox(facial_point_info[proc_num])
    
    new_h = h = image_data.shape[0]
    new_w = w = image_data.shape[1]
    channels  = image_data.shape[2]

    if h * w > MAX_HW:
      resize_ratio = math.sqrt(MAX_HW / (h * w))
      
      new_h = int(math.floor(h * resize_ratio))
      new_w = int(math.floor(w * resize_ratio))
      
      for idx in xrange(bbox_coord.size):
        bbox_coord[idx] = int(math.floor(bbox_coord[idx] * resize_ratio))
      
      for idx in xrange(facial_point_info[proc_num].size):
        facial_point_info[proc_num][idx] = \
            int(math.floor(facial_point_info[proc_num][idx] * resize_ratio))
      
      image_resize = imresize(image_data, (new_h, new_w, channels))
      imsave(os.path.join(dir_src,image_name), image_resize)
    
    image_mask = np.zeros((new_h,new_w), dtype='uint8') # not 'int'
    image_mask[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]] = label

    mat_data = {}
    #mat_data['name'] = image_name
    #mat_data['size'] = image_data.shape[:2]
    mat_data['points'] = facial_point_info[proc_num]
    mat_data['label'] = label
    mat_data['mask'] = image_mask
    mat_data['bbox'] = bbox_coord
    
    savemat(os.path.join(dir_dst,mat_name), mat_data, do_compression=True)
    
    #image_resize = imresize(image_data, (new_h, new_w, channels))
    #image_resize[:, 0:bbox_coord[0]] = 0
    #image_resize[:, bbox_coord[2]:w] = 0
    #image_resize[0:bbox_coord[1], :] = 0
    #image_resize[bbox_coord[3]:h, :] = 0
    #imsave(os.path.join(dir_dst,image_name), image_resize)
    
    if proc_num % 1000 == 0:
      print('%d / %d' % (proc_num, total_num_image))


if __name__ == '__main__':
  main()
