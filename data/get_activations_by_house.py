# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:12:33 2018

@author: Raymond
"""


import pickle
from nilmtk import DataSet, TimeFrame
import pandas as pd


redd = DataSet("data/redd.h5")
metadata = dict(redd.metadata)


# 将所有activations按房间号放入  
activations = {}
for i in range(1,7):
      activations[i] = {}  
for house in range(1,7):
      print('house_',house)
      elec = redd.buildings[house].elec
      appliances = elec.appliances
      for app in appliances:
            name = app.type['type']
            print(' '*5,name)
            actis = elec[name].get_activations()
            if actis != []:
                  activations[house][name] = [i.values for i in actis]

with open("activations_by_house.pickle", 'wb') as file:
      pickle.dump(activations, file)
print('已保存')
