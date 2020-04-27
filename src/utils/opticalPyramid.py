#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/16/2020
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

def layer(im1, im2, maxSpeed, winSize=5, eroRate=9, dilRate=9, minSpeed=1, horiParam=0.8, Euclian=False):
    opParam = dict(pyr_scale=0.5, levels=5, winsize=winSize, iterations=5, poly_n=1, poly_sigma=0, flags=0)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **opParam)
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else (horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1]))/2
    res = np.uint8(np.zeros_like(flow))
    if not maxSpeed:
        res[flow>minSpeed]=1 
    else:
        res[flow>minSpeed]=1
        res[flow>maxSpeed]=0
    res = cv2.erode(res, np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res, np.ones((dilRate,dilRate),np.uint8))
    score = np.sum(res)
    grid = np.mgrid[0:im1.shape[0], 0:im1.shape[1]]
    #mx = int(np.ceil(np.sum(grid[0]*res)/score)) if score else 0
    #my = int(np.ceil(np.sum(grid[1]*res)/score)) if score else 0
    return res, score

def opticalPyramid(im1, im2, minScore=100, maxSpeed=None):
    res, score = layer(im1, im2, maxSpeed)
    if score>minScore:
        x,y,w,h = cv2.boundingRect(res)
        if w*h<10000:
            x,y,w,h = 0, 0, 0, 0
    else:
        x,y,w,h = 0,0,0,0
    return (x,y,w,h)

if __name__=="__main__":
    # find object centre
    # find bounding box in ROI
    # Based on dense optical flow
    import utils.DataLoader
    path= '../raw_data/RectifiedImg/right/'
    d = DataLoader.DataLoader(path, size=(1024, 729), cvt=cv2.COLOR_BGR2RGB)
    for i in range(d.len):
        im1 = d.getItem(0)
        im2 = d.getItem(i)
        gray1, gray2 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        r= opticalPyramid.opticalPyramid(im1, im2, minScore=100, maxSpeed=2)
        im = im2.copy()
        cv2.rectangle(im, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0), 5)
        cv2.imshow('test', im)
        cv2.waitKey(1)
    cv2.destroyAllWindows()