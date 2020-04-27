#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/16/2020
'''
import cv2
import numpy as np
import math
import os
import glob

class DataLoader(object):
    def __init__(self, path, idx=0, cvt=None, size=None, downsize=None):
        super(DataLoader).__init__()
        self.path = path
        self.cvt = cvt
        self.idx = idx
        self.size = size
        self.downsize = downsize
        self.file = glob.glob(path+'*.png')
        self._len = len(self.file)

    def cvtImg(self, im):
        if isinstance(self.cvt, np.int):
            im = cv2.cvtColor(im, self.cvt)
        if isinstance(self.size, tuple):
            im = cv2.resize(im, self.size)
        if isinstance(self.downsize, np.int):
            new_size =  (int(im.shape[1]/self.downsize),int(im.shape[0]/self.downsize))
            im = cv2.resize(im, dsize=new_size)
        return im

    def getItem(self, idx):
        im = self.cvtImg(cv2.imread(self.file[idx]))
        return im
    
    def __next__(self):
        im = self.getItem(self.idx)
        self.idx+=1
        return im
    
    def getRest(self):
        return [self.__next__() for i in range(self.idx, self.len)]

    @property
    def len(self):
        return self._len

        
        




