# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:12:35 2018

@author: Raymond
"""

from nilmtk.dataset_converters import convert_redd
convert_redd('data/low_freq','redd.h5', format='HDF')
