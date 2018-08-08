# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:46:24 2018

@author: Raymond
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



import random as rn
rn.seed(12345)
import os
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow import set_random_seed
set_random_seed(2)    
# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)
            



from lib import get_house_raw_data,data_generator,segment
from lib import get_differential_sequence,remove_abnormal_points
from lib import get_modelPath,metrics_func,timer,randomly_insert
            
class Seq2Seq(object):
      
      def __init__(self,mode='o'):
            
            if mode == 'o':
                  self.name = 'Seq2Seq'
            if mode == 'od':
                  self.name = 'Seq2Seq_d'
            if mode == 'odd':
                  self.name = 'Seq2Seq_dd'
                  
            self.mode = mode
            self.trained = True
            
      def _load_activations(self,target_app):
            name_by_house, actis = pd.read_pickle('data/activations.pickle')
            other_names = []
            lengths = []
            for key,value in actis.items():
                  if key != target_app:
                        other_names.append(key)
                        lengths.append(len(value))
            total = np.sum(lengths)
            self.other_apps = other_names
            self.p = [i/total for i in lengths]
            
            from itertools import cycle
            self.activations = {key:cycle(value) for key,value in actis.items()}
            print('已加载 activations')
            
            
      def _get_odd_data(self,house):
            data = get_house_raw_data(house, self.appliance) 
            if data is None:
                return None
            else:
                X_o,y_o = data 
            X_o = remove_abnormal_points(X_o)  # 去除异常点
            y_o = remove_abnormal_points(y_o)  # 去除异常点
            X_o,y_o = X_o/self.MAX_X, y_o/self.MAX_y # (samples,1)
                  
            X_d = get_differential_sequence(X_o)  # (samples,)
            X_dd = get_differential_sequence(X_d) # (samples,)
            y_d = get_differential_sequence(y_o)  # (samples,)
            
            X = np.vstack((X_o,X_d,X_dd)).transpose()  # (samples,3)
            y = np.vstack((y_o,y_d)).transpose()       # (samples,3)
            
            return X,y 
              
      def get_house_data(self,
                         house, 
                         appliance, 
                         windows_length, 
                         MAX_X = 1,
                         MAX_y = 1):
            '''
            Single House
            '''
            self.house = house
            self.appliance = appliance
            self.windows_length = windows_length
            self.MAX_X = MAX_X
            self.MAX_y = MAX_y
            
            print('\n============ Loading data  ==============')
            
            
            data = self._get_odd_data(house)
            if data is None:
                print('\n该房间无该用电器，无法获取数据！')
                return
            else:
                XXX,yy = data
            assert XXX.shape[1]==3,yy.shape[1]==2 # X.shape=(样本数,3)

            from mylib import my_train_test_split
            self.X_train,self.X_test,self.y_train,self.y_test = my_train_test_split(XXX,yy,0.1)
            
            print('\nX_train.shape = {}'.format(self.X_train.shape))
            print('y_train.shape = {}'.format(self.y_train.shape))
            
            print('\nX_test.shape = {}'.format(self.X_test.shape))
            print('y_test.shape = {}'.format(self.y_test.shape))
            
            
      def get_houses_data(self, 
                          train_houses, 
                          test_house, 
                          appliance, 
                          windows_length, 
                          MAX_X = 1,
                          MAX_y = 1):   
            '''
            for unseen house
            '''
            self.appliance = appliance
            self.windows_length = windows_length
            self.MAX_X = MAX_X
            self.MAX_y = MAX_y
            
            print('\n-------- Load Training Data ---------')
            self._load_activations(appliance)
            first = True
            for house in train_houses:
                  # 提取一个房间的数据并生成三路odd数据
                  data = self._get_odd_data(house)
                  if data is None:
                      continue
                  else:
                      X_i,y_i = data
                  
                  # 合并房间i的数据
                  if first == True:
                        X_train = X_i # shape=(samples,3)
                        y_train = y_i # shape=(samples,2)
                        first = False
                  else:
                        # shape=(samples+,3)
                        X_train = np.vstack((X_train,X_i))
                        y_train = np.vstack((y_train,y_i))

            # 将所有房间数据并成总的train数据，然后直接可以喂给网络了
            self.X_train = X_train
            self.y_train = y_train
                        
            print('\n-------- Load Testing Data ---------')
            data = self._get_odd_data(test_house)
            if data is None:
                print('\n房间无该用电器，无法获取数据！')
                return
            else:
                X_i,y_i = data
                
            self.X_test = X_i   # shape=(samples,3)
            self.y_test = y_i   # shape=(samples,2)
            
            print('\nX_train.shape = {}'.format(self.X_train.shape))
            print('y_train.shape = {}'.format(self.y_train.shape))
            
            print('\nX_test.shape = {}'.format(self.X_test.shape))
            print('y_test.shape = {}'.format(self.y_test.shape))
            
           
      def build_network(self,
                        Type = 'Dense',
                        optimizer='adam', 
                        loss='mse',
                        plotModel = False):
          
            print('\n=============== Build a new Network  =================')
            from model import model
            self.train_number = 1
            self.Type = Type
            self.name, self.model = model(self,Type)
            r2,mae,sae = metrics_func(self)
            self.model.compile(optimizer=optimizer, 
                               loss=loss,
                               metrics=[r2,mae,sae]) # optimizer='adam'
            self.model.summary()
            
            ''' plot model ''' 
            if plotModel is True:
                  from keras.utils import plot_model
                  plot_model(self.model,to_file='model.png')
#                           rankdir='TB'  # TB for v or LR for h
                  

      
      def load_model_and_history(self,Type):
            print('\n=============== Load an exiting model  =================')
            from keras.models import load_model
            
            r2,mae,sae = metrics_func(self)
            print('Mode = ',self.mode,'\nType = ',Type)
            
            modelPath = get_modelPath()
            self.model = load_model(modelPath,{'r2':r2,'mae':mae,'sae':sae})
            print('已加载 model 文件：   {}'.format(modelPath))
            
            self.name = modelPath.split('\\')[1][:-3]
            resultPath = r'result\{}.history'.format(self.name)
            try:
                data = pd.read_pickle(resultPath)
                print('已加载 history 文件： {}\n'.format(resultPath))
                self.history = data
                self.train_number = 2
            except Exception as e:
                print('未加载 history 文件\n{}'.format(e))
                self.train_number = 1
            self.Type = self.name.split('(')[1].split(')')[0]

      @timer
      def train(self,epochs=5):
            print('\n============ Training  ==============')
            print('Model Name :',self.name)
            
            from keras.callbacks import EarlyStopping,LambdaCallback,CSVLogger,ModelCheckpoint
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
            logger = CSVLogger('result/{}.csv'.format(self.name), separator=',',append=True)
            check_point = ModelCheckpoint('result/best/{}.hdf5'.format(self.name), 
                                          verbose=1, monitor='val_r2',mode='max', 
                                          save_best_only=True,save_weights_only=False)
            self.trained = True
            
            train_generator = data_generator(self,data_set='train',p=0.5)
            valid_generator = data_generator(self,data_set='test',p=1)
            
            
            history = self.model.fit_generator(
                    train_generator, 
#                    steps_per_epoch = 90, 
                    steps_per_epoch = (len(self.X_train)//128+1)*2, 
                    epochs=epochs,
                    validation_data = valid_generator,
#                    validation_steps = 30,
                    validation_steps = len(self.X_test)//128,
                    verbose=1,
                    callbacks=[early_stopping,logger,check_point])

            if self.train_number == 1:
                  self.history = history.history
            elif self.train_number > 1:
                  for key in history.history.keys():
                        self.history[key].extend(history.history[key])
            self.train_number += 1
            
            
      def save_model_and_history(self):
            if self.trained == True:
                  modelPath = r'result/{}.h5'.format(self.name)
                  self.model.save(modelPath)
                  print('\n已保存model到  ',modelPath)
                  
                  import pickle
                  filepath = r'result/{}.history'.format(self.name)
                  with open(filepath, 'wb') as file:
                        pickle.dump(self.history, file)
                  print('已保存history到',filepath)

            else:
                  print('Error: 拜托，还没训练呢，保存个鬼啊！~')
          
            
            
            
          
      # for vis
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
      
      
      def evaluation_result(self,save=False):
            print('\n============  Evaluation  ==============')
            
            generator_train = data_generator(self,data_set='train',p=1)
            generator_test = data_generator(self,data_set='test',p=1)
            
            eval_train = self.model.evaluate_generator(
#                    generator_train,steps = 3,verbose=1)
                    generator_train,steps = len(self.X_train)//128,verbose=1)
            eval_test  = self.model.evaluate_generator(
#                    generator_test,steps = 3,verbose=1)
                    generator_test,steps = len(self.X_test)//128,verbose=1)
            
            eval_train = [round(i,4) for i in eval_train]
            eval_test = [round(i,4) for i in eval_test]
            
            self.result_train = {key:eval_train[i+1] for i,key in enumerate(['r2','mae','sae'])}
            self.result_test = {key:eval_test[i+1] for i,key in enumerate(['r2','mae','sae'])}
            
            if True:
                print('\nTraining:')
                print('SAE =',self.result_train['sae'])
                print('MAE =',self.result_train['mae'])
                print('R^2 =',self.result_train['r2'])
                        
                print('\nTesting:')
                print('SAE =',self.result_test['sae'])
                print('MAE =',self.result_test['mae'])
                print('R^2 =',self.result_test['r2'])
            
#            self._demo_show()
#            
#            if save == True:
#                  self._save_results()     
            
  
      def demo_show(self):
            print('\n-------- Demo result for show ----------')
                  
            X_train_seg, y_train_seg = segment(self.X_train,self.y_train,self.windows_length)
            X_test_seg, y_test_seg = segment(self.X_test,self.y_test,self.windows_length)
            
            if self.Type != 'Dense':
                  if len(X_train_seg[0].shape)==2:
                        X_train_seg = [np.expand_dims(i,axis=2) for i in X_train_seg]
                        X_test_seg = [np.expand_dims(i,axis=2) for i in X_test_seg]
            # 评估训练集
            X_train      = X_train_seg[0].reshape(-1,1)*self.MAX_X
            y_train      = y_train_seg[0].reshape(-1,1)*self.MAX_y
            y_train_pred = self.model.predict(X_train_seg[:len(self.mode)]).reshape(-1,1)*self.MAX_y
            
            # 评估测试集
            X_test       = X_test_seg[0].reshape(-1,1)*self.MAX_X
            y_test       = y_test_seg[0].reshape(-1,1)*self.MAX_y
            y_test_pred  = self.model.predict(X_test_seg[:len(self.mode)]).reshape(-1,1)*self.MAX_y

            if True:
                  from sklearn.metrics import r2_score

                  y_train_mean = np.mean(y_train)
                  y_train_pred_mean = np.mean(y_train_pred)
                  
                  result_train = {}
                  result_train['r2'] = round(r2_score(y_train, y_train_pred),4)
                  result_train['mae'] = round(np.mean(np.abs(y_train-y_train_pred)),2)
                  result_train['sae'] = round(np.abs(y_train_mean-y_train_pred_mean)/y_train_mean,4)
                  result_train['X'] = X_train 
                  result_train['y'] = y_train 
                  result_train['y_pred'] = y_train_pred
                  
                  
                  y_test_mean = np.mean(y_test)
                  y_test_pred_mean = np.mean(y_test_pred)
                  
                  result_test = {}
                  result_test['r2'] = round(r2_score(y_test, y_test_pred),4)
                  result_test['mae'] =round(np.mean(np.abs(y_test-y_test_pred)),2)
                  result_test['sae'] = round(np.abs(y_test_mean-y_test_pred_mean)/y_test_mean,4)
                  result_test['X'] = X_test 
                  result_test['y'] = y_test
                  result_test['y_pred'] = y_test_pred
                  
            if True:
                  print('\nDemo result for TrainingSet:')
                  print('SAE =',result_train['sae'])
                  print('MAE =',result_train['mae'])
                  print('R^2 =',result_train['r2'])
                        
                  print('\nDemo result for TestingSet:')
                  print('SAE =',result_test['sae'])
                  print('MAE =',result_test['mae'])
                  print('R^2 =',result_test['r2']) 
                
            if True:
                  plt.figure()
                        
                  plt.suptitle(self.name)
                  plt.subplot(2, 1, 1)
                  plt.title('Results on Training Set      ($R^2$ score={:.3f})'.format(result_train['r2']))
                  plt.plot(X_train,label='Aggregate Data')
                  plt.plot(y_train,label='({}) Ground Truth'.format(self.appliance))
                  plt.plot(y_train_pred,label='({}) Prediction'.format(self.appliance))
                  plt.legend(loc='upper right')
                  plt.ylabel('Power/W')
                        
                  plt.subplot(2, 1, 2)
                  plt.title('Results on Testing Set      ($R^2$ score={:.3f})'.format(result_test['r2']))
                  plt.plot(X_test,label='Aggregate Data')
                  plt.plot(y_test,label='({}) Ground Truth'.format(self.appliance))
                  plt.plot(y_test_pred,label='({}) Prediction'.format(self.appliance))
                  plt.legend(loc='upper right')
                  plt.xlabel('Time/s')
                  plt.ylabel('Power/W')
                        
                  plt.show()
            
                  
      def vis_layers(self):
            self.model.get_config
            self.model.get_output_at()
            self.model.get_config


                  
