#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/24/2020
'''

import requests
import os
from matplotlib import pyplot as plt
from selenium import webdriver
import re
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

'''
This is a script collecting images from https://www.pexels.com
'''

class Crawer(object):
    def __init__(self, word, number, save_dir='raw_data/imgs', url='https://www.pexels.com/search/'):
        super(Crawer, self).__init__()
        self.word = word # the keyword
        self.num = number # the minium number 
        self.idx = 0 # current featched image index
        self.url = url # search engine 
        self.browser = webdriver.Chrome()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os.chdir(save_dir)

    def connect(self,url):
        '''
        connect with the website and return a page
        '''
        self.browser.get(url)
    
    def search(self):
        '''
        search keywords and return a page
        '''
        word = '%20'.join(self.word.split(' '))+'/'
        url = self.url+word
        self.browser.get(url)
    
    @staticmethod
    def fetch(url):
        '''
        analyze the page and get the images
        '''
        img = requests.get(url).content
        return img

    def loopPage(self, write=True):
        html = self.browser.page_source
        imgs = re.findall(r'img srcset="([\s\S]+?) 1x', html)
        for i in range(len(imgs)):
            suffix = re.findall(r'images.pexels.com/photos/([\s\S]+?\?)', imgs[i])[0]
            suffix = re.findall(r'(\.[\s\S]+?)\?',suffix)[0]
            if write:
                self.save(self.fetch(imgs[i]), suffix)
            self.idx+=1
            print('Image '+str(self.idx)+'/'+str(self.num))
            if self.idx>=self.num:
                return True
        return False
        
    def scrollDown(self):
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        ActionChains(self.browser).key_down(Keys.DOWN).perform()
        time.sleep(3)
    
    def loop(self):
        self.search()
        for i in range(int(self.num//30)+1):
            self.scrollDown()
        while(True):
            if self.loopPage():
                break

    def save(self, img, suffix):
        '''
        save images in a local directory
        '''
        word = '_'.join(self.word.split(' '))
        name = word+f'{self.idx:04d}'+suffix
        with open(name, 'wb') as f:
            f.write(img)