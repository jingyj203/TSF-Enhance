
from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from distutils.dir_util import copy_tree
import io
import h5py
from shutil import copyfile
import time
import random
import cv2
class BaseDatasetMod(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.image_paths = []
        if not os.path.exists(root):
            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()

            copy_tree(source, root)

            elapsed = time.time() - c_time
            print('done copying file: %.2fs', elapsed)

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im_path = self.im_paths[index]
        im = PIL.Image.open(im_path)
        if len(list(im.split())) == 1 : im = im.convert('RGB') 
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index, im_path

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


