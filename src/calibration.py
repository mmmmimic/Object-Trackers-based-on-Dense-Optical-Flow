#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/07/2020
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import copy
import glob
import os

##
nb_horizontal = 9
nb_vertical = 6
# world coordinates, x->y->z
# set x and y coordinates
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
# set z coordinates
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

##
# first read images to calibrate
left = glob.glob("raw_data/Stereo_calibration_images/left*.png")
right = glob.glob("raw_data/Stereo_calibration_images/right*.png")
imgpl = []
imgpr = []
img_left = []
img_right = []
objpoints = []
for i in range(len(left)):
    iml = cv2.imread(left[i])
    imr = cv2.imread(right[i])
    retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
    imgl = copy.deepcopy(iml)
    imgr = copy.deepcopy(imr)
    cv2.drawChessboardCorners(imgl, (nb_vertical,nb_horizontal), corners_l, retl)
    cv2.drawChessboardCorners(imgr, (nb_vertical,nb_horizontal), corners_r, retr)
    # when and only when both of the left and right images in a pair has the corners
    if retl and retr: 
        imgpl.append(corners_l)
        imgpr.append(corners_r)
        img_left.append(iml)
        img_right.append(imr)
        objpoints.append(objp)
    #cv2.imshow('left', imgl)
    #cv2.waitKey(5)
    #cv2.imshow('right', imgr)
    #cv2.waitKey(10)
#cv2.destroyAllWindows()

# determine the camera matirx 1
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints, imgpl, (imgl.shape[:2])[::-1], None, None)
# determine the camera matirx 2
_, mtx2, dist2,_ ,_ = cv2.calibrateCamera(objpoints, imgpr, (imgr.shape[:2])[::-1], None, None)

# undistortion
def undistort(img, mtx, dist, root):
    counter = 0
    if not os.path.exists(root):
        os.makedirs(root)
    for im in img:
        im = cv2.imread(im)
        dst = cv2.undistort(im, mtx, dist)
        cv2.imwrite(root+'/'+str(counter)+'.png', dst)
        counter = counter+1

left = glob.glob("raw_data/Stereo_conveyor_with_occlusions/left/*.png")
right = glob.glob("raw_data/Stereo_conveyor_with_occlusions/right/*.png")
root = 'raw_data/UndistortedImgOcc/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 

left = glob.glob("raw_data/Stereo_conveyor_without_occlusions/left/*.png")
right = glob.glob("raw_data/Stereo_conveyor_without_occlusions/right/*.png")
root = 'raw_data/UndistortedImg/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 

# undistort checkboard
left = glob.glob("raw_data/Stereo_calibration_images/left*.png")
right = glob.glob("raw_data/Stereo_calibration_images/right*.png")
root = 'raw_data/UndistortedCheckBoard/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 

## calibrate via the undistorted checkboard
left = glob.glob("raw_data/UndistortedCheckBoard/left/*.png")
right = glob.glob("raw_data/UndistortedCheckBoard/right/*.png")
imgpl = []
imgpr = []
objpoints = []
for i in range(len(left)):
    iml = cv2.imread(left[i])
    imr = cv2.imread(right[i])
    retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
    imgl = copy.deepcopy(iml)
    imgr = copy.deepcopy(imr)
    cv2.drawChessboardCorners(imgl, (nb_vertical,nb_horizontal), corners_l, retl)
    cv2.drawChessboardCorners(imgr, (nb_vertical,nb_horizontal), corners_r, retr)
    # when and only when both of the left and right images in a pair has the corners
    if retl and retr: 
        imgpl.append(corners_l)
        imgpr.append(corners_r)
        objpoints.append(objp)
    #cv2.imshow('left', imgl)
    #cv2.waitKey(5)
    #cv2.imshow('right', imgr)
    #cv2.waitKey(10)
#cv2.destroyAllWindows()
# determine the camera matirx 1
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints, imgpl, (imgl.shape[:2])[::-1], None, None)
# determine the camera matirx 2
_, mtx2, dist2,_ ,_ = cv2.calibrateCamera(objpoints, imgpr, (imgr.shape[:2])[::-1], None, None)

# stereo calibration
_, mtx1, dist1, mtx2, dist2, R, T,_ ,_ = cv2.stereoCalibrate(objpoints, imgpl, imgpr, mtx1, dist1, mtx2, dist2, (imgl.shape[:2])[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

# camera information
cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2, 'R':R, 'T':T}
with open('raw_data/cameraInfo.txt','w') as f:
    f.write(str(cameraInfo))
