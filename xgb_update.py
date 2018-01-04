#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximum Entropy Model
"""
import pandas as pd
import pickle
import gzip
import xgboost as xgb
import gc

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

x_train, x_valid, y_train, y_valid=load('xgb_trainset.pc')
labels=load('xgb_labels.pc')


#%% Xgboost

num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
gc.collect()
param = {'objective':'multi:softmax',
         'eta':'0.3', 'max_depth':10,
         'silent':0, 
         'num_class':num_class,
         'eval_metric':'merror'}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                  verbose_eval=10,xgb_model='xgb.model')
model.save_model('xgb.model')

