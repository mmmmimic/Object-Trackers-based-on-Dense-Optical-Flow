#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/19/2020
'''
from utils import Tracker
import cv2

'''
GREEN box means "Object observed"
RED box means "Target observed"
YELLOW box means "Target under occlusion"
'''

path= 'data/imgs'
size = (1024, 729)
monsize = 20 # the target path under occulusion is predicted in line with the previous 20 frames
thresh = [10, 5]
kernel = (200, 200)
stride = (30, 30)

def localTrack():
    sampling = True
    t = Tracker.LocalDenseTracker(path, None, None, monsize, thresh, kernel, stride, sampling=True, rec=False)
    t.loop(0, Filter=True)

def globalTrack():
    t = Tracker.GlobalDenseTracker(path, None, None, monsize, thresh, False)
    t.loop(0,Filter=True)

def BSTrack():
    trainFrame = 60 # the number of frames used for training
    t = Tracker.BSTracker(path, None, None, monsize, thresh, trainFrame, False)
    t.loop(0,Filter=True)

if __name__=='__main__':
    #localTrack()
    #globalTrack()
    BSTrack()