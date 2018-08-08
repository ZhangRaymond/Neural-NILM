# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:10:52 2018

@author: Raymond
"""
from keras.layers import Input,Dense,concatenate,multiply 
from keras.layers import Dropout,BatchNormalization,Reshape
from keras.models import Model
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten,Activation,Lambda
from keras.layers.advanced_activations import ELU,PReLU,LeakyReLU




def model(seq2seq,Type):
    
    
    l = seq2seq.windows_length
    mode = seq2seq.mode
    model_name = seq2seq.name
    model_name += '({})'.format(Type)
    
    if Type == 'CNN(Chaoyun)':
        
        o = Input(shape=(l,1))
        
        x = Conv1D(30,(10), strides=1,activation='relu')(o)
        x = MaxPooling1D(2)(x)
        x = Conv1D(30,(8), strides=1,activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(40,(6), strides=1,activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(50,(5), strides=1,activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(50,(5), strides=1,activation='relu')(x)
        x = MaxPooling1D(2)(x)
                
        f = Flatten()(x)
        f = Dense(1024,activation='relu')(f)   
        output = Dense(l, activation='relu')(f)
            
            
            
            
    if Type == 'Dense':
        
        o = Input(shape=(l,))
        d = Input(shape=(l,))
        dd = Input(shape=(l,))
                
        if mode == 'o':
            x = o
        if mode == 'od':
            x = concatenate([o,d])
        if mode == 'odd':
            x = concatenate([o,d,dd])

#                      x = Dropout(0.2)(x)
#                      x = BatchNormalization()(x)  
                   
        x = Dense(2*l, activation='relu')(x)
        x = Dense(2*l, activation='relu')(x)
        x = Dense(2*l, activation='relu')(x)
        x = Dense(2*l, activation='relu')(x)
        x = Dense(l, activation='relu')(x)
        x = Dense(l, activation='relu')(x)
        output = Dense(l, activation='relu')(x)
            
            
            
    if Type == 'CNN-1d-2':
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
                
        x_d = Conv1D(10,(5), strides=1,activation='relu')(d)
        x_d = MaxPooling1D(2)(x_d)
        x_d = Conv1D(10,(5), strides=1,activation='relu')(x_d)
        x_d = MaxPooling1D(2)(x_d)
        x_d = Conv1D(10,(5), strides=1,activation='relu')(x_d)
        x_d = MaxPooling1D(2)(x_d)
        x_d = Conv1D(5,(5), strides=1,activation='relu')(x_d)
        x_d = MaxPooling1D(2)(x_d)
        
        f = Flatten()(x)
        f_d = Flatten()(x_d)
        c = concatenate([f,f_d])
                
        if mode == 'o':
            out = f
        elif mode == 'od':
            out = c
                    
        out    = Dense(2*l, activation='relu')(out)   
        output = Dense(l, activation='relu')(out)
        
    if Type == 'CNN-multi-kernel':
        o = Input(shape=(l,1))
        d = Input(shape=(l,1))
        dd = Input(shape=(l,1))
        
        x1 = Conv1D(20,(3),strides=1,activation='relu')(o)
        x1 = MaxPooling1D(2)(x1)
        x1 = Conv1D(30,(3), strides=1,activation='relu')(x1)
        x1 = MaxPooling1D(2)(x1)
        x1 = Conv1D(40,(3), strides=1,activation='relu')(x1)
        x1 = MaxPooling1D(2)(x1)
        x1 = Conv1D(50,(3), strides=1,activation='relu')(x1)
        x1 = MaxPooling1D(2)(x1)

        
        x2 = Conv1D(20,(5),strides=1,activation='relu')(o)
        x2 = MaxPooling1D(2)(x2)
        x2 = Conv1D(30,(5), strides=1,activation='relu')(x2)
        x2 = MaxPooling1D(2)(x2)
        x2 = Conv1D(40,(5), strides=1,activation='relu')(x2)
        x2 = MaxPooling1D(2)(x2)
        x2 = Conv1D(50,(5), strides=1,activation='relu')(x2)
        x2 = MaxPooling1D(2)(x2)

        
        x3 = Conv1D(20,(10),strides=1,activation='relu')(o)
        x3 = MaxPooling1D(2)(x3)
        x3 = Conv1D(30,(10), strides=1,activation='relu')(x3)
        x3 = MaxPooling1D(2)(x3)
        x3 = Conv1D(40,(10), strides=1,activation='relu')(x3)
        x3 = MaxPooling1D(2)(x3)
        x3 = Conv1D(50,(10), strides=1,activation='relu')(x3)
        x3 = MaxPooling1D(2)(x3)

        

                
                
        f1 = Flatten()(x1)
        f2 = Flatten()(x2)
        f3 = Flatten()(x3)
        c = concatenate([f1,f2,f3])
        
                
#        if mode == 'o':
#            out = f
#        elif mode == 'od':
#            out = c
                    
        out    = Dense(2*l, activation='relu')(c)   
        output = Dense(l, activation='relu')(out)
                
                
                
    if Type == 'CNN-2d':
        
        o = Input(shape=(l,1))
        d = Input(shape=(l,1))
        dd = Input(shape=(l,1))
                
        o_d = concatenate([o,d],axis=-1)
        x = Conv1D(30,(5), strides=1,activation='linear')(o_d)
        x = MaxPooling1D(2)(x)
        x = Conv1D(30,(5), strides=1,activation='linear')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(40,(3), strides=1,activation='linear')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(50,(3), strides=1,activation='linear')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(50,(5), strides=1,activation='linear')(x)
        x = MaxPooling1D(2)(x)
                
        f = Flatten()(x)
        out    = Dense(l, activation='relu')(f)   
        output = Dense(l, activation='relu')(out)
                
                
                
    if Type == 'RNN':
        from keras.layers.wrappers import Bidirectional
        from keras.layers.embeddings import Embedding
                
        o = Input(shape=(l,1))
        d = Input(shape=(l,1))
        dd = Input(shape=(l,1))
        
        x = Conv1D(20,(5), strides=1,activation='linear')(o)
        x = Conv1D(10,(10), strides=1,activation='linear')(x)
        x = Conv1D(1,(10), strides=1,activation='linear')(x)
                
        x = Conv1D(10,(10), strides=1,activation='linear')(o)
        x = Bidirectional(LSTM(20,activation='relu',return_sequences=True),
                          merge_mode='concat')(o)
#                x = LSTM(10,activation='tanh',return_sequences=True)(o)
        x = Dense(1,activation='relu')(x)
        output = Flatten()(x)
        
    if Type == 'CNN-RNN':
        from keras.layers.wrappers import Bidirectional
        from keras.layers.embeddings import Embedding
                
        o = Input(shape=(l,1))
        d = Input(shape=(l,1))
        dd = Input(shape=(l,1))
        
        x = Conv1D(20,(5),padding='same', strides=1,activation='relu')(o)
#        x = Conv1D(10,(10), padding='same',strides=1,activation='linear')(x)
        x = Conv1D(10,(10), padding='same',strides=1,activation='relu')(x)
                
        x = LSTM(20,activation='relu',return_sequences=True)(x)
        x = LSTM(1,activation='relu',return_sequences=True)(x)
        output = Flatten()(x)
    
    if Type == 'ConvLSTM':
        from keras.layers import ConvLSTM2D
        
        o = Input(shape=(l,1))
        d = Input(shape=(l,1))
        dd = Input(shape=(l,1))
                
        
        x = ConvLSTM2D(10, kernel_size=(5,1), strides=(1, 1),activation='relu')(o)
        x = Dense(1,activation='relu')(x)
        output = Flatten()(x)
        
        
        
        

            
    if mode == 'o':
        model = Model(o, output)
    if mode == 'od':
        model = Model([o,d], output)
    if mode == 'odd':
        model = Model([o,d,dd], output)
                  
    return model_name,model