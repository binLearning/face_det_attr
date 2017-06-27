import os
import sys
import shutil
from six.moves import xrange


def main():
  dir_src = sys.argv[1]
  
  dir_train = os.path.join(dir_src, 'train')
  dir_val   = os.path.join(dir_src, 'val')
  dir_test  = os.path.join(dir_src, 'test')
  
  os.mkdir(dir_train)
  os.mkdir(dir_val)
  os.mkdir(dir_test)
  
  file_name_form = '{:0>6}.' + sys.argv[2] # jpg or mat
  
  # train        1 - 162770 (162770)
  # val     162771 - 182637 (19867)
  # test    182638 - 202599 (19962)
  pos_train = 162771
  pos_val   = 182638
  pos_end   = 202599 + 1
  for idx in xrange(1, pos_train):
    file_name = file_name_form.format(idx)
    path_src = os.path.join(dir_src, file_name)
    path_dst = os.path.join(dir_train, file_name)
    shutil.move(path_src, path_dst)
    
  for idx in xrange(pos_train, pos_val):
    file_name = file_name_form.format(idx)
    path_src = os.path.join(dir_src, file_name)
    path_dst = os.path.join(dir_val, file_name)
    shutil.move(path_src, path_dst)
  
  for idx in xrange(pos_val, pos_end):
    file_name = file_name_form.format(idx)
    path_src = os.path.join(dir_src, file_name)
    path_dst = os.path.join(dir_test, file_name)
    shutil.move(path_src, path_dst)
    

if __name__ == '__main__':
  main()
