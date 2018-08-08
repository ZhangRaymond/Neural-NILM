# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:24:57 2018

@author: Raymond

"""

import random as rn
rn.seed(12345)
import os
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.optimizers import RMSprop,Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from packages import *


def get_house_raw_data(house,appliance):
            print('house_{}:'.format(house))
            data = pd.read_pickle('data/house_{}.pickle'.format(house))
            device_name = data.deviceName
            device_index = None
            for i,device in enumerate(device_name):
                  if device.endswith(appliance):
                        device_index = i
                        break
            if device_index == None:
                  print('        Does not have this appliance: {}'.format(appliance))
                  return
            else: 
                  print('        appliance: {}   Index = {}'.format(appliance,device_index))
                  X = data.mains
                  y = data.appliances[:,device_index]
            return X,y
      
#def get_house_raw_data(house,appliance):
#            print('house_{}:'.format(house))
#            raw,data = pd.read_pickle('data/house_{}_manually.pickle'.format(house))
#            device_name = data.device_name
#            device_index = None
#            for i,device in enumerate(device_name):
#                  if device.endswith(appliance):
#                        device_index = i-2
#                        break
#            if device_index == None:
#                  print('        Does not have this appliance: {}'.format(appliance))
#                  return
#            else: 
#                  print('        appliance: {}   Index = {}'.format(appliance,device_index))
#                  Mains = raw.mains['mains']
#                  Appliance = raw.appliances.iloc[:,device_index]
#                  All = pd.concat([Mains, Appliance], axis=1).dropna(axis=0,how='any')
#                  X = All.iloc[:,0].values
#                  y = All.iloc[:,1].values
#            return X,y

def segment(X,y,seg_length):
      l = X.shape[0]%seg_length #序列长度（times）
      if l==0: 
            l=-1  
      X_seg = [np.reshape(X[:,i][:-l],(-1,seg_length)) for i in range(3)]
      y_seg = [np.reshape(y[:,i][:-l],(-1,seg_length)) for i in range(2)]
                  
      return X_seg, y_seg
            
def shift_segment(X,y,seg_length,stride,print_info=True):
      '''
      X is X_o-X_d-X_dd  shape = (samples,3)
      y is y_o-y_d       shape = (samples,2)
      '''
      X_o_seg = []
      X_d_seg = []
      X_dd_seg = []
      y_o_seg = []
      y_d_seg = []
      
      for i in range(len(X)-seg_length+1):
            if i%stride==0:
                  assert len(X[i:i+seg_length,0]) == seg_length
                  X_o_seg.append(  X[i:i+seg_length,0] ) 
                  X_d_seg.append(  X[i:i+seg_length,1] )
                  X_dd_seg.append( X[i:i+seg_length,2] )
                  
                  y_o_seg.append(  y[i+seg_length//2-1,0] )
                  y_d_seg.append(  y[i+seg_length//2-1,1] )
      if print_info==True:           
            print(' '*7,'sequence length = {}'.format(len(X[:,0])))
            print(' '*7,'windows length = {}'.format(seg_length))
            print(' '*7,'stride = {}'.format(stride))
            print(' '*7,'segments =',len(y_o_seg))
      # (segments,seg_length)
      return [np.array(X_o_seg),np.array(X_d_seg), np.array(X_dd_seg)], [np.array(y_o_seg),np.array(y_d_seg)]


def segment_generator():
      pass

def power_to_on_off(y):
      pass

def get_differential_sequence(X):      
      X_d = []
      pre_item = 0
      for item in X:
            X_d.append(item-pre_item)
            pre_item = item
      X_d = np.array(X_d) # shape = (samples,)
      return X_d

def recover_sequence(d):
      r = [d[0]]
      pre_item = d[0]
      for item in d[1:]:
            pre_item = item+pre_item
            r.append(pre_item)
      return np.array(r)


def remove_abnormal_points(y,left_threshold=200,right_threshold=80):
      y_ = []
      for i,value in enumerate(y):
      
            if i == 0 or i==len(y)-1:
                  y_.append(y[i])
            else:
                  if y[i]-y[i-1]>left_threshold and y[i]-y[i+1]>right_threshold:
#                        print('index:',i)
                        y_.append(y[i+1]+1)
                  else:
                        y_.append(y[i])
      return np.array(y_)

def get_modelPath():
      import glob
      files = glob.glob(r'trained_models/*.h5')
      fileNum = len(files)
      if fileNum == 0:
            raise LookupError(' 注意：该文件夹下没有model.h5文件！')
      if fileNum == 1:
            print('\n当前文件夹下检测到以下model.h5文件:')
            print(r'   {}'.format(files[0]))
            i = input('确定载入？ [y]/n    ')
            if i == 'n':
                  raise Exception('放弃载入模型，Game over！')
            fileName = files[0]
            return fileName
      elif fileNum > 0:
            print('\n当前文件夹下检测到以下model.h5文件:')
            for i in range(fileNum):
                 print('  {}  {}'.format(i+1,files[i]))
            select = input('请选择要处理的文件[1-{}]: '.format(fileNum))
            fileName = files[int(select)-1]
            return fileName



class Seq2point(object):
      
      def __init__(self,mode='o'):
            
            if mode == 'o':
                  self.name = 'Seq2point'
            if mode == 'od':
                  self.name = 'Seq2point_d'
            if mode == 'odd':
                  self.name = 'Seq2point_dd'
                  
            self.mode = mode
#            self.train_number = 1
            
      def get_odd_data(self,house):
            X_o,y_o = get_house_raw_data(house, self.appliance) 
            
            X_o = remove_abnormal_points(X_o)  # 去除异常点
            X_o = X_o/self.MAX_X # (samples,1)
            X_d = get_differential_sequence(X_o)  # (samples,)
            X_dd = get_differential_sequence(X_d) # (samples,)
            
            y_o = remove_abnormal_points(y_o)  # 去除异常点
            y_o = y_o/self.MAX_y # (samples,1)
            y_d = get_differential_sequence(y_o)  # (samples,)
            
            X = np.vstack((X_o,X_d,X_dd)).transpose()  # (samples,3)
            y = np.vstack((y_o,y_d)).transpose()       # (samples,3)
            
            return X,y 
              
      def get_house_data(self,
                         house, 
                         appliance, 
                         windows_length, 
                         stride,
                         MAX_X = 1,
                         MAX_y = 1):
            '''
            Single House
            '''
            self.house = house
            self.appliance = appliance
            self.windows_length = windows_length
            self.stride = stride
            
            self.MAX_X = MAX_X
            self.MAX_y = MAX_y
            
            print('\n============ Loading data  ==============')
            
            XXX,yy = self.get_odd_data(house)
            assert XXX.shape[1]==3,yy.shape[1]==2 # X.shape=(样本数,3)

            from mylib import my_train_test_split
            self.X_train,self.X_test,self.y_train,self.y_test = my_train_test_split(XXX,yy,0.1)
            
            self.X_train_seg,self.y_train_seg = shift_segment(self.X_train,self.y_train,
                                                              self.windows_length,stride)
            self.X_test_seg,self.y_test_seg = shift_segment(self.X_test,self.y_test,
                                                            self.windows_length,stride)

            print('\nX_train.shape = {}'.format(self.X_train.shape))
            print('y_train.shape = {}'.format(self.y_train.shape))
            
            print('X_o_train_seg.shape = {}'.format(self.X_train_seg[0].shape))
            print('y_o_train_seg.shape = {}'.format(self.y_train_seg[0].shape))
            
            print('\nX_test.shape = {}'.format(self.X_test.shape))
            print('y_test.shape = {}'.format(self.y_test.shape))
            
            print('X_o_test_seg.shape = {}'.format(self.X_test_seg[0].shape))
            print('y_o_test_seg.shape = {}'.format(self.y_test_seg[0].shape))
            
            
            
      def get_houses_data(self, 
                          train_houses, 
                          test_house, 
                          appliance, 
                          windows_length, 
                          stride,
                          MAX_X = 1,
                          MAX_y = 1):   
            '''
            for unseen house
            '''
            self.appliance = appliance
            self.windows_length = windows_length
            self.stride = stride
            
            self.MAX_X = MAX_X
            self.MAX_y = MAX_y
            
            
            print('-------- Load Training Data ---------')
            first = True
            for house in train_houses:
                  # 提取一个房间的数据并生成三路odd数据
                  X_i,y_i = self.get_odd_data(house=house)
                  X_seg_i,y_seg_i = shift_segment(X_i,y_i,self.windows_length,self.stride)
                  
                  # 合并房间i的数据
                  if first == True:
                        
                        X_train = X_i # shape=(samples,3)
                        y_train = y_i # shape=(samples,2)
                        
                        # shape=(samples,seg_length)
                        X_o_train_seg,X_d_train_seg,X_dd_train_seg = X_seg_i
                        y_o_train_seg,y_d_train_seg = y_seg_i
                        
                        first = False
                  else:
                        # shape=(samples+,3)
                        X_train = np.vstack((X_train,X_i))
                        y_train = np.vstack((y_train,y_i))
                        
                        # shape=(samples+,seg_length)
                        X_o_train_seg = np.vstack((X_o_train_seg,X_seg_i[0]))
                        X_d_train_seg = np.vstack((X_d_train_seg,X_seg_i[1]))
                        X_dd_train_seg = np.vstack((X_dd_train_seg,X_seg_i[2]))
                        y_o_train_seg = np.hstack((y_o_train_seg,y_seg_i[0]))
                        y_d_train_seg = np.hstack((y_d_train_seg,y_seg_i[1]))
            
            # 将所有房间数据并成总的train数据，然后直接可以喂给网络了
            self.X_train = X_train
            self.y_train = y_train
            self.X_train_seg = [X_o_train_seg,X_d_train_seg,X_dd_train_seg]
            self.y_train_seg = [y_o_train_seg,y_d_train_seg ]

                        
            print('-------- Load Testing Data ---------')
            
            X_i,y_i = self.get_odd_data(house=test_house)
            X_seg_i,y_seg_i = shift_segment(X_i,y_i,self.windows_length,self.stride)
            self.X_test = X_i   # shape=(samples,3)
            self.y_test = y_i   # shape=(samples,2)
            self.X_test_seg = X_seg_i 
            self.y_test_seg = y_seg_i
            
            
            print('\nX_train.shape = {}'.format(self.X_train.shape))
            print('y_train.shape = {}'.format(self.y_train.shape))
            
            print('X_o_train_seg.shape = {}'.format(self.X_train_seg[0].shape))
            print('y_o_train_seg.shape = {}'.format(self.y_train_seg[0].shape))
            
            print('\nX_test.shape = {}'.format(self.X_test.shape))
            print('y_test.shape = {}'.format(self.y_test.shape))
            
            print('X_o_test_seg.shape = {}'.format(self.X_test_seg[0].shape))
            print('y_o_test_seg.shape = {}'.format(self.y_test_seg[0].shape))
            
            
            
            
      def build_network(self,
                        Type = 'Dense',
                        optimizer='adam', 
                        loss='mse',
                        dropout = False,
                        BN = False):
            print('\n=============== Build a new Network  =================')
            self.train_number = 1
            
 
            from keras.layers import Input,Dense,concatenate,multiply 
            from keras.layers import Dropout,BatchNormalization,Reshape
            from keras.models import Model
            from keras.layers.recurrent import LSTM,GRU
            from keras.layers.convolutional import Conv1D
            from keras.layers.pooling import MaxPooling1D
            from keras.layers.core import Flatten,Activation,Lambda
            
            self.Type = Type
            l = self.windows_length
            import keras.backend as K
            def gaussian(x):
                  return K.exp(-(x-K.ones_like(x))**2)
             
            
            if Type == 'CNN(Chaoyun)':
                self.name = self.name+'(CNN-Chaoyun)'
                
                o = Input(shape=(l,1))
                
                x = Conv1D(30,(10), strides=1,activation='relu')(o)
#                x = MaxPooling1D(2)(x)
                x = Conv1D(30,(8), strides=1,activation='relu')(x)
#                x = MaxPooling1D(2)(x)
                x = Conv1D(40,(6), strides=1,activation='relu')(x)
#                x = MaxPooling1D(2)(x)
                x = Conv1D(50,(5), strides=1,activation='relu')(x)
#                x = MaxPooling1D(2)(x)
                x = Conv1D(50,(5), strides=1,activation='relu')(x)
#                x = MaxPooling1D(2)(x)
                
                f = Flatten()(x)
                f = Dense(1024,activation='relu')(f)   
                output = Dense(1, activation='relu')(f)
            
            
            
            
            if Type == 'Dense':
                self.name = self.name+'(Dense)'
                o = Input(shape=(l,))
                d = Input(shape=(l,))
                dd = Input(shape=(l,))
                
                if self.mode == 'o':
                      x = o
                if self.mode == 'od':
                      x = concatenate([o,d])
                if self.mode == 'odd':
                      x = concatenate([o,d,dd])

#                      x = Dropout(0.2)(x)
#                      x = BatchNormalization()(x)  
                   
                x = Dense(2*l, activation='relu')(x)
                x = Dense(2*l, activation='relu')(x)
                x = Dense(2*l, activation='relu')(x)
                x = Dense(2*l, activation='relu')(x)
                x = Dense(l, activation='relu')(x)
                x = Dense(l, activation='relu')(x)
                output = Dense(1, activation='relu')(x)
            
            
            if Type == 'CNN':
                self.name = self.name+'(CNN)'
                o = Input(shape=(l,1))
                d = Input(shape=(l,1))
                dd = Input(shape=(l,1))
                
                x = Conv1D(30,(5), strides=1,activation='relu')(o)
                x = MaxPooling1D(2)(x)
                x = Conv1D(30,(5), strides=1,activation='relu')(x)
                x = MaxPooling1D(2)(x)
                x = Conv1D(40,(3), strides=1,activation='relu')(x)
                x = MaxPooling1D(2)(x)
                x = Conv1D(50,(3), strides=1,activation='relu')(x)
                x = MaxPooling1D(2)(x)
                x = Conv1D(50,(5), strides=1,activation='relu')(x)
                x = MaxPooling1D(2)(x)
                
                
                x_d = Conv1D(10,(3), strides=1,activation='linear')(d)
                x_d = MaxPooling1D(2)(x_d)
                x_d = Conv1D(10,(3), strides=1,activation='linear')(d)
#                x_d = Activation(gaussian)(x_d)
                x_d = MaxPooling1D(2)(x_d)
                x_d = Conv1D(10,(3), strides=1,activation='linear')(x_d)
                x_d = MaxPooling1D(2)(x_d)
                x_d = Conv1D(5,(3), strides=1,activation='linear')(x_d)
                x_d = MaxPooling1D(2)(x_d)
                
                
                f = Flatten()(x)
                
                f_d = Flatten()(x_d)
                c = concatenate([f,f_d])
                
                if self.mode == 'o':
                      out = f
                elif self.mode == 'od':
                      out = c
                    

                out    = Dense(l,activation='relu')(out)   
                output = Dense(1, activation='relu')(out)
                
                
            if Type == 'RNN':
                self.name = self.name+'(RNN)'
                o = Input(shape=(l,1))
                d = Input(shape=(l,1))
                dd = Input(shape=(l,1))
                
                x=LSTM(100,activation='tanh')(o)
                output = Dense(l,activation='relu')(x)
            
            
            
            if self.mode == 'o':
                  self.model = Model(o, output)
            if self.mode == 'od':
                  self.model = Model([o,d], output)
            if self.mode == 'odd':
                  self.model = Model([o,d,dd], output)
                  

                  
            def r2(y_true, y_pred):
                  SS_res =  K.sum(K.square(y_true - y_pred)) 
                  SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
                  return 1 - SS_res/(SS_tot) # + K.epsilon()
            def mae(y_true, y_pred):
                  return K.mean(K.abs(y_pred-y_true))*self.MAX_y
            def sae(y_true, y_pred):
                  tot_gt = K.sum(y_true)
                  tot_pr = K.sum(y_pred)
                  return K.abs(tot_gt-tot_pr)/(tot_gt)
            def modified_loss(y_true, y_pred):
                mse = K.mean(K.pow(K.abs(y_pred-y_true),3), axis=-1)
#                punishment = K.mean(K.dot(K.abs(y_pred - y_true),K.transpose(y_true))/100, axis=-1)
                punishment = 0
                return mse+punishment

            if loss == "modified_loss":
                loss = modified_loss
            self.model.compile(optimizer=optimizer, loss=loss,
                               metrics=[r2,mae,sae]) # optimizer='adam'
            print(self.model.summary())
            
#            from keras.utils import plot_model
#            plot_model(self.model,
#                       to_file='model.png'
##                       rankdir='TB'  # TB for v or LR for h
#                       )
      def power_to_on_off(self,y):
          if y>(100/self.MAX_y):
              return 1
          else:
              return 0  
            
            
            
      def load_model(self):
            print('\n=============== Load an exiting model  =================')
            import keras
            import keras.backend as K
            def r2(y_true, y_pred):
                  SS_res =  K.sum(K.square(y_true - y_pred)) 
                  SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
                  return 1 - SS_res/(SS_tot) # + K.epsilon()
            def mae(y_true, y_pred):
                  return K.mean(K.abs(y_pred-y_true))*self.MAX_y
            def sae(y_true, y_pred):
                  tot_gt = K.sum(y_true)
                  tot_pr = K.sum(y_pred)
                  return K.abs(tot_gt-tot_pr)/(tot_gt)
            
            print('Mode = ',self.mode)
            modelPath = get_modelPath()
            self.model = keras.models.load_model(modelPath,{'r2':r2,'mae':mae,'sae':sae})
            print('已加载 h5 文件：     {}'.format(modelPath))
                  
            self.name = modelPath.split('\\')[1][:-3]
            resultPath = r'resultData\{}.pickle'.format(self.name)
            data = pd.read_pickle(resultPath)
            print('已加载 pickle 文件： {}\n'.format(resultPath))
            self.history = data.train_history
            self.train_number = 2
            self.Type = self.name.split('(')[1].split(')')[0]
            
      def train(self,epochs=20):
            print('\n============ Training  ==============')
            
            from keras.callbacks import EarlyStopping, ModelCheckpoint
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
            #filepath = 'trained_models/model-weight-best.hdf5'
            #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
            #                             save_best_only=True,save_weights_only=False,mode='min')
            if self.Type in ['CNN','RNN','CNN(Chaoyun)']:
                  if len(self.X_train_seg[0].shape)==2:
                        self.X_train_seg = [np.expand_dims(i,axis=2) for i in self.X_train_seg]
                        self.X_test_seg = [np.expand_dims(i,axis=2) for i in self.X_test_seg]
                  
            i = len(self.mode) 
            X_train_seg = self.X_train_seg[:i]  # 'o'取第一个数据， 'od'取前两个数据,'odd'取前三个（全部）数据
            y_train_seg = self.y_train_seg[0]   # y_o_train_seg
            
            self.history_i = self.model.fit(X_train_seg, 
                                            y_train_seg, 
                                            epochs=epochs,
                                            batch_size=128, 
                                            validation_split=0.005,
#                                          validation_data=(self.X_test_seg[:i],
#                                                           self.y_test_seg[0]),
                                            shuffle=True,
                                            verbose=1,
                                            callbacks=[early_stopping])
            
            filepath = r'trained_models/{}.h5'.format(self.name)
            self.model.save(filepath)
            print('\n已保存model到',filepath)

            if self.train_number == 1:
                  self.history = self.history_i.history
            elif self.train_number > 1:
                  for key in self.history_i.history.keys():
                        self.history[key].extend(self.history_i.history[key])
            self.train_number = self.train_number + 1      
            
            
            #weights = model.get_weights()
            #model.load_weights(filepath)
            #best_weight = model.get_weights()
            
      def plot_training_history(self):
            print('\n============ Plot Training history  ==============')
            plt.figure()
            for key in self.history.keys():
                  if key in ['loss','val_loss']:
                        plt.subplot(2,2,1)
                        plt.plot(self.history[key],label=key)
                        plt.ylabel('Mean Square Error')
                        plt.xlabel('epoch')
                        plt.legend(loc='upper right')
                  elif key in ['mae','val_mae']:
                        plt.subplot(2,2,2)
                        plt.plot(self.history[key],label=key)
                        plt.ylabel('Mean Absolute Error')
                        plt.xlabel('epoch')
                        plt.legend(loc='upper right')
                  elif key in ['sae','val_sae']:
                        plt.subplot(2,2,3)
                        plt.plot(self.history[key],label=key)
                        plt.ylabel('Signal Aggregate Error')
                        plt.xlabel('epoch')
                        plt.legend(loc='upper right')
                  elif key in ['r2','val_r2']:
                        plt.subplot(2,2,4)
                        plt.plot(self.history[key],label=key)
                        plt.ylabel('$R^2$ score')
                        plt.xlabel('epoch')
                        plt.legend(loc='best')
            plt.show()
      
      def predict_one_by_one(self,X,y):
            '''
            X is  shape = (samples,3)
            y is  shape = (samples,2)
            '''
            seg_length = self.windows_length
            print(' '*7,'sequence length = {}'.format(len(X)))
            print(' '*7,'windows length = {}'.format(seg_length))
            
            X_o = X[:,0]
            y_o = y[:,0]

            X_o_ = []
            y_pred = []
            y_true = []

            L = len(X)-seg_length+1
            l = 2000
            import time
            start_time = time.time()
            for i in range(L):
                  if L-i>L%l:
                        X_o_.append(X_o[i+seg_length//2-1])
                        y_true.append(y_o[i+seg_length//2-1])
                  
                  if i>0 and i%l==0:
                        X_seg,y_seg = shift_segment(X[i-l:i+seg_length-1],y[i-l:i+seg_length-1],seg_length,stride=1,print_info=False)
                        if self.Type in ['CNN','RNN','CNN(Chaoyun)']:
                              if len(X_seg[0].shape)==2:
                                    X_seg = [np.expand_dims(i,axis=2) for i in X_seg]
                                    y_seg = [np.expand_dims(i,axis=2) for i in y_seg]
                        y_pred.append(np.squeeze(self.model.predict(X_seg[:len(self.mode)])))
                        timing = round(time.time()-start_time,1)
                        print('\r        {}/{}   {}%    Timing:{}s'.format(i,L,round(i/L*100,1),timing),end='', flush=True)
            
            y_pred = np.squeeze(np.array(y_pred).reshape(-1,1)) 
            print('') # 取消 flush 的影响
#            print(len(X_o_))
#            print(len(y_pred))
            assert len(X_o_) == len(y_pred) and len(X_o_) == len(y_true)
            
            X_o_ = np.array(X_o_)*self.MAX_X
            y_true = np.array(y_true)*self.MAX_y
            y_pred = np.array(y_pred)*self.MAX_y
            
            return X_o_, y_true, y_pred
      
      def release_memory_of_segs_data(self):
            
            self.X_train_seg = 0
            self.y_train_seg = 0
            
            self.X_test_seg = 0
            self.y_test_seg = 0
            
      def simulation_result(self,
                            print_result = False,
                            plot_result = False,
                            save = False):
#            print('\n============ Simulation Result  ==============')
            print('\n-------- {} ----------'.format(self.name))

            # X_o and y_o
            print('\nPredicting on training set:')
            X_train, y_train,y_train_pred = self.predict_one_by_one(self.X_train,self.y_train)
            print('\nPredicting on testing set:')
            X_test, y_test, y_test_pred = self.predict_one_by_one(self.X_test,self.y_test)

           
            if True:
                  from sklearn.metrics import r2_score

                  y_train_mean = np.mean(y_train)
                  y_train_pred_mean = np.mean(y_train_pred)
                  
                  self.result_train = {}
                  self.result_train['r2'] = round(r2_score(y_train, y_train_pred),4)
                  self.result_train['mae'] = round(np.mean(np.abs(y_train-y_train_pred)),2)
                  self.result_train['sae'] = round(np.abs(y_train_mean-y_train_pred_mean)/y_train_mean,4)
                  self.result_train['y_mean'] = round(y_train_mean,2)
                  self.result_train['y_pred_mean'] = round(y_train_pred_mean,2)
                  
                  
                  y_test_mean = np.mean(y_test)
                  y_test_pred_mean = np.mean(y_test_pred)
                  
                  self.result_test = {}
                  self.result_test['r2'] = round(r2_score(y_test, y_test_pred),4)
                  self.result_test['mae'] =round(np.mean(np.abs(y_test-y_test_pred)),2)
                  self.result_test['sae'] = round(np.abs(y_test_mean-y_test_pred_mean)/y_test_mean,4)
                  self.result_test['y_mean'] = round(y_test_mean,2)
                  self.result_test['y_pred_mean'] = round(y_test_pred_mean,2)
                  
                  
                  if print_result == True:
                        print('\nTraining:')
                        print('SAE       =',self.result_train['sae'])
                        print('MAE       =',self.result_train['mae'])
                        print('R^2       =',self.result_train['r2'])
                        print('GT_mean   =',self.result_train['y_mean'])
                        print('pred_mean =',self.result_train['y_pred_mean'])
                        
                        print('\nTesting:')
                        print('SAE       =',self.result_test['sae'])
                        print('MAE       =',self.result_test['mae'])
                        print('R^2       =',self.result_test['r2'])
                        print('GT_mean   =',self.result_test['y_mean'])
                        print('pred_mean =',self.result_test['y_pred_mean'])
                  
                  
                  if plot_result == True:
                        plt.figure()
                        
                        plt.suptitle(self.name)
                        plt.subplot(2, 1, 1)
                        plt.title('Results on Training Set      ($R^2$ score={:.3f})'.format(self.result_train['r2']))
                        plt.plot(X_train,label='Aggregate Data')
                        plt.plot(y_train,label='({}) Ground Truth'.format(self.appliance))
                        plt.plot(y_train_pred,label='({}) Prediction'.format(self.appliance))
                        plt.legend(loc='upper right')
                        plt.ylabel('Power/W')
                        
                        plt.subplot(2, 1, 2)
                        plt.title('Results on Testing Set      ($R^2$ score={:.3f})'.format(self.result_test['r2']))
                        plt.plot(X_test,label='Aggregate Data')
                        plt.plot(y_test,label='({}) Ground Truth'.format(self.appliance))
                        plt.plot(y_test_pred,label='({}) Prediction'.format(self.appliance))
                        plt.legend(loc='upper right')
                        plt.xlabel('Time/s')
                        plt.ylabel('Power/W')
                        
                        plt.show()       
                        
                  if save == True:
                        from sklearn.datasets import base 
                        import pickle
                        
                        data = base.Bunch(name = self.name,
                                          appliance = self.appliance,
                                          
                                          X_train = X_train,
                                          y_train = y_train,
                                          y_train_pred = y_train_pred,
                                          result_train = self.result_train,
                                          
                                          X_test = X_test,
                                          y_test = y_test,
                                          y_test_pred = y_test_pred,
                                          result_test = self.result_test,
                                          
                                          train_history = self.history
                                          )
                        
                        resultPath = r'resultData/{}.pickle'.format(self.name)
                        with open(resultPath, 'wb') as file:
                              pickle.dump(data, file)
                        print('\n已保存result到',resultPath)
                  
                  
