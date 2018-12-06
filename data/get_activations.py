# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:12:33 2018

@author: Raymond
"""


import pickle
import pandas as pd

activations = pd.read_pickle('activations_by_house.pickle')

# get all appliance name in all houses
appNames = set()
for house in range(1,7):
      for i in activations[house].keys():
            appNames.add(i)
appNames = list(appNames)


# load activations of each appliance in all houses
appNames_house = {}  # appName --- [1,2,5]  有这个app的房间
appNames_actis = {}  # appName --- activations   这个App的所有activations
for name in appNames:
      appNames_house[name] = []
      appNames_actis[name] = []
      for house in range(1,7):
            if name in list(activations[house].keys()) and house!=1 :
                  appNames_house[name].append(house)
                  appNames_actis[name].extend(activations[house][name])

with open("activations.pickle", 'wb') as file:
      pickle.dump([appNames_house,appNames_actis], file)
print('saved.')

