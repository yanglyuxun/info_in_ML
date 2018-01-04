#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nltk
"""
import nltk
from nltk.stem import WordNetLemmatizer
import zipfile
import pandas as pd
import numpy as np
#import pickle
from collections import Counter
#import gzip
import random
import sklearn
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.metrics import *
from sklearn.pipeline import Pipeline

save = lambda o,f:pd.io.pickle.to_pickle(o,f,'gzip')
load = lambda f:pd.io.pickle.read_pickle(f,'gzip')

stat=pd.io.pickle.read_pickle('stat.pc','gzip')
#train=pd.io.pickle.read_pickle('train.pc','gzip')
trains,tests=pd.io.pickle.read_pickle('train_test.pc','gzip')

#%% add context
def addshift(df,n,name):
    df1=df.shift(n)
    df[name]=df1.before.fillna('')
    wrong_ind=(df.sentence!=df1.sentence)
    df[name].loc[wrong_ind]=''
    return df
train=addshift(train,1,'tm1')
train=addshift(train,-1,'tp1')
train=train[['tm1','before','tp1','cls','after','sentence', 'token']]
save(train,'train.pc')

#%% resample
n_need=stat.n.iloc[2:].sum()
ind=train.query('cls=="PLAIN" or cls=="PUNCT"').sample(n_need).index
train.cls[ind]="PLAIN" # no need to seperate them # changed train!!!!!
ind2=train.query('cls!="PLAIN" and cls!="PUNCT"').index
ind=ind.append(ind2)
del ind2
ind_other = [i for i in train.index if i not in ind]
train_new = train.loc[ind]
train_other = train.loc[ind_other]
train_other.cls='PLAIN'
del train

#%% get features

def features(row):
    f={}
    for i,c in enumerate(row[1]):
        f['t'+str(i)]=c
        if c.isspace():
            l='space'
            f[l]=f.get(l,0)+1
        elif c=='.':
            l='dot'
            f[l]=f.get(l,0)+1
        elif c==',':
            l='comma'
            f[l]=f.get(l,0)+1
        elif c=='-':
            l='minus'
            f[l]=f.get(l,0)+1
        elif c==':':
            l='twodot'
            f[l]=f.get(l,0)+1
        elif c=='/':
            l='slash'
            f[l]=f.get(l,0)+1
        elif c.islower():
            l='low'
            f[l]=f.get(l,0)+1
        elif c.isupper():
            l='up'
            f[l]=f.get(l,0)+1
        elif c.isdigit():
            l='dig'
            f[l]=f.get(l,0)+1
        else:
            l='other'
            f[l]=f.get(l,0)+1
        let=f.get('low',0)+f.get('up',0)
        if let:
            f['letter']=let
    for i,c in enumerate(row[0]):
        f['tm'+str(i)]=c
    for i,c in enumerate(row[2]):
        f['tp'+str(i)]=c
    return (f,row[3])
def set_sep(sets0,test_frac=0.1):
    sets = sets0.copy() 
    random.shuffle(sets)
    n = int(test_frac*len(sets))
    return sets[n:], sets[:n]
dataset=[features(row) for row in train_new.as_matrix()]
random.seed(777)
trains,tests=set_sep(dataset)
del train_new,train_m
del dataset
otherset=[features(row) for row in train_other.sample(frac=0.1,random_state=777).as_matrix()]
tests+=otherset
random.seed(777)
random.shuffle(tests)
pd.io.pickle.to_pickle([trains,tests],'train_test.pc','gzip')

#%%#############################################
#%% classif
class my_clssifier():
    def __init__(self, skclssif):
        self.skclssif=skclssif
        self.classif=nltk.classify.scikitlearn.SklearnClassifier(self.skclssif)
        try:
            self.classif._clf.set_params(n_jobs=-1)
        except:
            pass
        self.pip=Pipeline([('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
                     #('chi2', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=1000)),
                     ('NB',skclssif)])
        self.pipclassif=nltk.classify.scikitlearn.SklearnClassifier(self.pip)
    def fit(self, train, test):
        self.classifier=self.classif.train(train)
        print('train finished!')
        print(nltk.classify.accuracy(self.classifier, test))
    def pipfit(self, train, test):
        self.pipclassifier=self.pipclassif.train(train)
        print('train finished!')
        print(nltk.classify.accuracy(self.pipclassifier, test))


#%% max ent
cl=my_clssifier(sklearn.linear_model.LogisticRegression(
        verbose=10,solver='saga',multi_class='multinomial'))
cl.fit(trains, tests)
pd.io.pickle.to_pickle(cl,'max_ent_model.pc','gzip')
cl.pipfit(trains, tests)
pd.io.pickle.to_pickle(cl,'max_ent_model.pc','gzip')


cl_maxent=pd.io.pickle.read_pickle('max_ent_model.pc','gzip')
#nltk.classify.accuracy(cl_maxent.classifier, trains)
nltk.classify.accuracy(cl_maxent.pipclassifier, trains)
#nltk.classify.accuracy(cl_maxent.classifier, tests)
nltk.classify.accuracy(cl_maxent.pipclassifier, tests)
0.9908060138489521

#%% MLPClassifier
from sklearn.neural_network import MLPClassifier
cl=my_clssifier(MLPClassifier(verbose=True,
                              hidden_layer_sizes=(20,15,10),
                              warm_start=True))
#cl.fit(trains, tests)
cl.pipfit(trains, tests)
pd.io.pickle.to_pickle(cl,'MLP_model.pc','gzip')
0.9940867717100673
'''
Iteration 1, loss = 0.19338066
Iteration 2, loss = 0.06334318
Iteration 3, loss = 0.05186207
Iteration 4, loss = 0.04535529
Iteration 5, loss = 0.04127996
Iteration 6, loss = 0.03820640
Iteration 7, loss = 0.03601182
Iteration 8, loss = 0.03413185
Iteration 9, loss = 0.03274547
Iteration 10, loss = 0.03152830
Iteration 11, loss = 0.03048794
Iteration 12, loss = 0.02957852
Iteration 13, loss = 0.02864295
Iteration 14, loss = 0.02797937
Iteration 15, loss = 0.02724539
'''

#%% AdaBoost
from sklearn.ensemble import AdaBoostClassifier
cl=my_clssifier(AdaBoostClassifier(n_estimators=50,algorithm='SAMME'))
cl.pipfit(trains, tests)
pd.io.pickle.to_pickle(cl,'AdaBoost_model.pc','gzip')
0.9576395078258274
#nltk.classify.accuracy(cl.pipclassifier, trains)


#cl=my_clssifier(AdaBoostClassifier(n_estimators=100))
#cl.pipfit(trains, tests)
#0.08452438085021435
#nltk.classify.accuracy(cl.pipclassifier, trains)
#0.08452438085021435
#pd.io.pickle.to_pickle(cl,'AdaBoost100_model.pc','gzip')

#%% Some base lines
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
cl=my_clssifier(MultinomialNB())
cl.pipfit(trains, tests)
0.9838805295994129
#SVM
#cl=my_clssifier(sklearn.svm.SVC())
#cl.pipfit(trains, tests)

# KNN
#cl=my_clssifier(sklearn.neighbors.KNeighborsClassifier())
#cl.pipfit(trains, tests)
