import caffe

import numpy as np
from PIL import Image
from scipy.io import loadmat

import random


class FaceDataLayer(caffe.Layer):
    """
    Load (input image, label mat) pairs from CelebA dataset,
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to CelebA dataset dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        - clip: select a part of data

        for FCN FACE detection and attribute recogniton.

        example

        params = dict(data_dir="/path-to-data",
            mean=(93.5940,104.7624,129.1863), #BGR
            split="train")
        """
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        #self.random = params.get('randomize', True)
        #self.seed = params.get('seed', None)
        self.clip = params.get('clip', 200)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        # CelebA dataset split:
        # train        1 - 162770  (162770)
        # val     162771 - 182637  (19867)
        # test    182638 - 202599  (19962)
        if self.split == 'train':
          self.indices = range(1, 162771)
        if self.split == 'val':
          self.indices = range(162771, 182638)[:self.clip]
        if self.split == 'test':
          self.indices = range(182638, 202560)
        
        self.idx = 0
        
        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False
        # randomization: seed and pick
        #if self.random:
        #    random.seed(self.seed)
        #    self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        
        self.idx += 1
        if self.idx == len(self.indices):
            self.idx = 0

        # pick next input
        #if self.random:
        #    self.idx = random.randint(0, len(self.indices)-1)
        #else:
        #    self.idx += 1
        #    if self.idx == len(self.indices):
        #        self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img_resize_1000/{}/{:0>6}.jpg'.format(self.data_dir, self.split, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        mat = loadmat('{}/info_resize_1000/{}/{:0>6}.mat'.format(self.data_dir, self.split, idx))
        label = mat['mask'][...].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label
