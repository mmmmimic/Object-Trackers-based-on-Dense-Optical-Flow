#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@author: mxl
@date: 04/25/2020
'''
from utils import Crawer
key_word = 'white cat'
num = 200
c = Crawer.Crawer(key_word, num, 'raw_data/imgs')
c.loop()



