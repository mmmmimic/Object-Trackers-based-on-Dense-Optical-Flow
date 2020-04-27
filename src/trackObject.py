#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/19/2020
'''
from utils import Tracker
import cv2

'''
GREEN box means "I find a moving object, but I'm not sure it is the target"
RED box means "I'm sure that this is the target"
YELLOW box means "I found the target and now it is under occlusion, so this is the position I guess based on the history"
'''

path= 'raw_data/RectifiedImg/right/'
size = (1024, 729) # unnecessary
monsize = 20 # the target path under occulusion is predicted based on the previous 20 frames
thresh = [10, 5]
kernel = (200, 200)
stride = (30, 30)

def localTrack():
    sampling = True # if True the fps is much higher, but the video will lose some frames
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