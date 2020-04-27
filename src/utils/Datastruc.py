#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/19/2020
'''
import numpy as np
from numpy import linalg
import random
import math
import scipy.signal as signal
from matplotlib import pyplot as plt

class Queue(object):
    def __init__(self, size):
        super(Queue, self).__init__()
        self.size = size
        self.data = []
    
    def push(self, d):
        if len(self.data)<self.size:
            self.data.append(d)
        else: 
            self.pop()
            self.data.append(d)

    def pop(self):
        return self.data.pop(0)

    def item(self, idx):
        return self.data[idx]

class Monitor(Queue):
    def __init__(self, size):
        super(Monitor, self).__init__(size)
        self.M = {'Regression':'regression', 'Mean':'mean', 'Gaussian':'guess', 'Robust':'robustMean', 'Same':'same'}

    @staticmethod
    def diff(data):
        assert isinstance(data, np.ndarray)
        dif = np.zeros((data.shape[0]-1, 2))
        for i in range(dif.shape[0]):
            dif[i,0] = data[i+1, 0] - data[i, 0]
            dif[i,1] = data[i+1, 1] - data[i, 1]
        return dif
    
    @staticmethod
    def regression(data):
        data = data.reshape(1,-1)
        u, w, vt = linalg.svd(data)
        p = np.abs(1-vt[-1,:])
        return np.dot(p, data.T)[0]
    
    @staticmethod
    def guess(data):
        # guess based on Gaussian distribution 
        return random.gauss(np.mean(data), np.std(data))
    
    @staticmethod
    def mean(data):
        return np.mean(data)

    @staticmethod
    def same(data):
        return data[-1]

    @staticmethod
    def robustMean(data, rate=0.7):
        return rate*np.mean(data)+(1-rate)*(random.gauss(np.mean(data), np.std(data)))

    @staticmethod
    def preprocess(data):
        mean = np.mean(data)
        std = np.std(data)
        data = data[data>=mean-3*std]
        data = data[data<=mean+3*std]
        return data
    
    @staticmethod
    def update(displacement, t, vel, acc):
        x_dis = displacement[-t,0]+vel[0]*t+0.5*acc[0]*t**2
        y_dis = displacement[-t,1]+vel[1]*t+0.5*acc[1]*t**2
        return np.array([[x_dis],[y_dis]])

    def process(self,data, method):
        ans = []
        for i in range(2):
            for j in range(2):
                tmp = self.preprocess(data[i][:,j])
                eval('ans.append(self.'+method+'(tmp))')
        return ans
        
    def predict(self, kernelSize, method):
        method = self.M[method]
        assert method
        data = np.array(self.data).reshape(-1,2)
        vel = self.diff(data)  
        acc = self.diff(vel)
        return self.update(data, 1, self.process([vel, acc], method)[:2], self.process([vel, acc], method)[2:])

class Footprint(object):
    def __init__(self):
        super(Footprint, self).__init__()
        self.data = []
        self.delta = 0
        self.status = 0
        self.var = []
        self.sx = 0
        self.sy = 0

    def push(self, point, status):
        self.status = status
        if self.isNewObj(point) and (status==3 or status==1):
            self.data.append(point)

    def motionModel(self, order, x, point, t):
        time = [int(i*t/order) for i in range(1, order+1)]
        S = np.array([self.getDist(point[list(x).index(min(x)), :2], 
        point[list(x).index(sorted(x)[t]), :2]) for t in time]).reshape(-1,1)
        T = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                T[i,j] = (time[i])**(j+1)/(j+1)
        var = linalg.pinv(T)@S
        return var

    def update(self):
        # update the motion model variables
        order = len(self.var)
        for i in range(order-1):
            for j in range(i+1, order):
                self.var[i]+=self.var[j]/(j)

    def move(self):
        # return delta_s
        order = len(self.var)
        delta_s = 0
        for i in range(order):
            delta_s += (self.var[i])/(i+1)
        return delta_s
             
    def regression(self, order=5):
        # return the predicted delta_x and delta_y, vel, acc and beta
        t = len(self.data)-1
        assert t
        point = np.array([np.array([data[0][0],data[1][0], 1]) for data in self.data]).reshape(-1, 3)
        u, w, vt = linalg.svd(point)
        # denoising: median filter, kernel_size:5
        point[:,0] = signal.medfilt(point[:,0], 5)
        point[:,1] = signal.medfilt(point[:,1], 5)
        v = vt[-1, :]
        #y = point[:,1]
        #assert v[0]
        #x = -(v[1]*y+v[2])/v[0]
        x = point[:,0]
        assert v[1]
        y = -(v[0]*x+v[2])/v[1]
        # see the fitting line
        #plt.plot(y,-x)
        #plt.show()
        if order==1:
            dx = (max(x)-min(x))/t
            dy = (max(y)-min(y))/t
            theta = math.atan(dy/dx)
            var = [np.sqrt(dx**2+dy**2)]
        else:
            var = self.motionModel(order, x, point, t)
            theta = math.atan(-v[0]/v[1])
        symbol_x = x[-1]-x[0]
        symbol_y = y[-1]-y[0]
        symbol_x/=np.abs(symbol_x) if symbol_x else 0
        symbol_y/=np.abs(symbol_y) if symbol_y else 0
        self.sx = symbol_x
        self.sy = symbol_y
        self.var = var
        self.theta = theta

    def predict(self, point):
        if len(self.data):
            self.regression()
        delta_s = self.move()
        dx, dy = np.abs(delta_s*math.cos(self.theta)), np.abs(delta_s*math.sin(self.theta))
        dx *= self.sx
        dy *= self.sy
        point[0] = int(point[0]+dx)
        point[1] = int(point[1]+dy)
        self.update()
        return point

    @staticmethod
    def getDist(start, stop):
        # get the distance between two points format::[1x2]ndarray
        return np.sqrt(((start-stop)@(start-stop).T))
        
    def isNewObj(self, point, metric=100):
        if len(self.data)>1:
            # metric: Euclian distance
            self.delta = self.getDist(point.T, self.data[-1].T)
            if self.delta>metric:
                self.data = []
                return False
        if self.status == 2:
            self.data = []
            return False
        return True
