# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:22:25 2018

@author: Raymond
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split


def randomly_insert(a, w):
    l = len(a)
    n = np.random.randint(-l+10, w-10)
    # print('w= {}\nl= {}\nn= {}'.format(w,l,n))
    if n < 0:
        if l+n < w:
            acti = np.append(a[-n:], np.zeros(w-(l+n)))
        else:
            acti = a[-n:(w-n)]
    else:
        if n+l < w:
            acti = np.append(np.append(np.zeros(n), a), np.zeros(w-(n+l)))
        else:
            acti = np.append(np.zeros(n), a[:w-n])

    assert len(acti) == w
    return acti



def randomly_select_appName(obj):
    # 按概率取其他用电器的数量
    num_others = np.random.choice(
        [1, 2, 3, 4], size=1, p=[0.4, 0.35, 0.15, 0.10])
    other_apps = np.random.choice(
        obj.other_apps, size=num_others, replace=False, p=obj.p)
    return other_apps


def generate_synthetic_data(obj):
    if obj.appliance == 'refrigerator':
        target_app = 'fridge'
    elif obj.appliance == 'washer_dryer':
        target_app = 'washer dryer'

    w = obj.windows_length
    # 生成target_app，出现的概率是0.6
    if np.random.rand() > 0.4:
        acti = next(obj.activations[target_app])
        target_acti = randomly_insert(acti, w)
    else:
        target_acti = np.zeros(w)

    # 生成other_apps，即bg
    other_apps = randomly_select_appName(obj)
    bg = np.zeros(w)
    for i in other_apps:
        acti = next(obj.activations[i])
        other_acti = randomly_insert(acti, w)
        bg += other_acti

    # 合成agg
    agg = target_acti+bg

    agg /= obj.MAX_X
    target_acti /= obj.MAX_y

    return agg, target_acti


def data_generator(obj, data_set, p):
    '''
    Args:
        obj:      the model object
        data_set: generate the data form this data_set 
        p:        probability of that the data comes from the existing agg data.
                    Should be given manually. Suggestion:
                        -- p = 0.5 for training
                        -- p = 1   for evaluation
    Return:     
        X : o-d-dd  shape = (samples,3)
        y : o-d     shape = (samples,2)
    '''
    assert 0 <= p <= 1
    from itertools import cycle
    netName = obj.netName
    mode = obj.mode
    w = obj.windows_length
    batch_size = obj.batch_size

    if data_set == 'train':
        X = obj.X_train
        y = obj.y_train
    elif data_set == 'test':
        X = obj.X_test
        y = obj.y_test
    indices = np.arange(0, len(X)-w+1)
    np.random.shuffle(indices)
    indices = cycle(indices)

    while True:
        X_o_seg = []
        X_d_seg = []
        # X_dd_seg = []
        y_o_seg = []
        # y_d_seg = []

        nums1 = int(batch_size*p)
        nums2 = batch_size - nums1

        for _ in range(nums1):
            i = next(indices)
            X_seg = X[i:i+w]
            y_seg = y[i:i+w]
            X_o_seg.append(X_seg[:, 0])
            X_d_seg.append(X_seg[:, 1])
            # X_dd_seg.append( X_seg[:,2] )
            y_o_seg.append(y_seg[:, 0])
            # y_d_seg.append( y_seg[:,1] )

        for _ in range(nums2):
            X_o, y_o = generate_synthetic_data(obj)
            X_d = get_differential_sequence(X_o)
            # X_dd = get_differential_sequence(X_d)
            # y_d = get_differential_sequence(y_o)

            X_o_seg.append(X_o)
            X_d_seg.append(X_d)
            # X_dd_seg.append(X_dd)
            y_o_seg.append(y_o)
            # y_d_seg.append(y_d)

        assert len(X_o_seg) == len(X_d_seg) == batch_size
        assert len(y_o_seg) == batch_size
        if mode == 'od':
            X_ = [np.array(X_o_seg),
                  np.array(X_d_seg)]
        elif mode == 'o':
            X_ = np.array(X_o_seg)
        elif mode == 'd':
            X_ = np.array(X_d_seg)
        # elif mode == 'odd':
        #            X_ = [np.array(X_o_seg),
        #                  np.array(X_d_seg),
        #                  np.array(X_dd_seg)]

        # 如果不是FCN，则需要增加一个维度
        if netName != 'FCN':
            if mode in ['o', 'd']:
                X_ = np.expand_dims(X_, axis=2)
            else:
                X_ = [np.expand_dims(i, axis=2) for i in X_]
        y_ = np.array(y_o_seg)
        yield [X_, y_]


def get_house_raw_data(house, appliance):
    data = pd.read_pickle('data/house_{}.pickle'.format(house))
    device_name = data.deviceName
    device_index = None
    for i, device in enumerate(device_name):
        if device.endswith(appliance):
            device_index = i
            break
    if device_index == None:
        print('house_{}: No appliance: {}'.format(house, appliance))
        return None
    else:
        print('house_{}: appliance: {}   Index = {}'.format(
            house, appliance, device_index))
        X = data.mains
        y = data.appliances[:, device_index]
        return X, y

def my_train_test_split(X, y,test_size=0.15):
    ''' 
    just split the last tst_size proportion of data 
    as testing set and others as the training set
    '''
    l = int(X.shape[0]*(1-test_size))
    X_train,y_train = X[:l], y[:l]
    X_test,y_test = X[l:], y[l:]
    return X_train, X_test, y_train, y_test

def segment(X, y, seg_length):
    l = X.shape[0] % seg_length  # 序列长度（times）
    if l == 0:
        l = -1
    X_seg = [np.reshape(X[:, i][:-l], (-1, seg_length))
             for i in range(X.shape[1])]
    y_seg = [np.reshape(y[:, i][:-l], (-1, seg_length))
             for i in range(y.shape[1])]

    return X_seg, y_seg


def seconds2min(seconds):
    hours = seconds//3600
    seconds = seconds % 3600
    mins = seconds//60
    seconds = seconds % 60
    return hours, mins, seconds


def get_differential_sequence(X):
    X_d = np.diff(X)
    return np.insert(X_d, obj=0, values = 0)


def remove_abnormal_points(y, left_threshold=200, right_threshold=80):
    y_ = []
    for i, value in enumerate(y):
        if i == 0 or i == len(y)-1:
            y_.append(y[i])
        else:
            if y[i]-y[i-1] > left_threshold and y[i]-y[i+1] > right_threshold:
                y_.append(y[i+1]+1)
            else:
                y_.append(y[i])
    return np.array(y_)


def get_modelPath(dir_path):
    import glob
    files = glob.glob(r'{}/*.h5'.format(dir_path))
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
            print('  {}  {}'.format(i+1, files[i]))
        select = input('请选择要处理的文件[1-{}]: '.format(fileNum))
        fileName = files[int(select)-1]
        return fileName


def metrics(obj):
    import keras.backend as K

    def cum_loss():
        ''' custom loss'''
        pass

    def r2(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res/(SS_tot)  # + K.epsilon()

    def mae(y_true, y_pred):
        return K.mean(K.abs(y_pred-y_true))*obj.MAX_y

    def sae(y_true, y_pred):
        tot_gt = K.sum(y_true)
        tot_pr = K.sum(y_pred)
        return K.abs(tot_gt-tot_pr)/(tot_gt)
    return r2, mae, sae


def timer(func):
    def wrapper(*args, **kw):
        import datetime
        starttime = datetime.datetime.now()

        func(*args, **kw)  # run function here

        endtime = datetime.datetime.now()
        print('\n训练开始时间：', starttime)
        print('训练结束时间：', endtime)
        total_time = (endtime - starttime).seconds
        h, m, s = seconds2min(total_time)
        print('训练时长： {}h {}m {}s   ({}s)'.format(h, m, s, total_time))
    return wrapper
