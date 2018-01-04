#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximum Entropy Model
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import nltk
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

#%%
stat=load('stat.pc')
train=load('train.pc')
X=load('X.pc')
#test=load('test.pc')
#sample=load('sample.pc')

#%% resample
n_need=stat.n.iloc[2:].sum()
ind=train.query('cls=="PLAIN" or cls=="PUNCT"').sample(n_need).index
train.cls[ind]="PLAIN" # no need to seperate them # changed train!!!!!
ind2=train.query('cls!="PLAIN" and cls!="PUNCT"').index
ind=ind.append(ind2)
del ind2
Xnew = X.loc[ind]
del X
ynew=pd.factorize(train.cls[ind].tolist()[0])
labels = ynew[1]
ynew = ynew[0]
# split
x_train, x_valid, y_train, y_valid= train_test_split(Xnew, ynew,
        test_size=0.1, random_state=2017)
del train,Xnew,ynew

#%% Xgboost

num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
gc.collect()
param = {'objective':'multi:softmax',
         'eta':'0.3', 'max_depth':10,
         'silent':0, 'nthread':4,
         'num_class':num_class,
         'eval_metric':'merror'}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                  verbose_eval=10,xgb_model='xgb.model')

model.save_model('xgb.model')
save([x_train, x_valid, y_train, y_valid],'xgb_trainset.pc')
save(labels,'xgb_labels.pc')

# then use google cloud to train many many rounds
# get the model back and check the accuracy
x_train, x_valid, y_train, y_valid=load('xgb_trainset.pc')
model=xgb.Booster()
model.load_model('xgb.model')# read model and see the accuracy
labels=load('xgb_labels.pc')

choosed_ind=x_train.index.append(x_valid.index)
else_X=X.drop(choosed_ind).sample(frac=0.1)
Xtest=x_valid.append(else_X).sort_index()
ytest=train.cls.loc[Xtest.index]
Xtest=Xtest.as_matrix()
del choosed_ind,else_X
del x_train, x_valid, y_train, y_valid
del X
gc.collect()
dXtest=xgb.DMatrix(Xtest)
del Xtest
ytestp=model.predict(dXtest)
ytestpl=[labels[int(i)] for i in ytestp]
ytest=ytest.replace('PUNCT','PLAIN')
ytest=pd.DataFrame(ytest)
ytest['pcls']=ytestpl
ytest['correct']=ytest.eval('cls==pcls')
ytest['correct'].mean() # error rate
err=ytest.query('correct==False')
err=err.join(train[['before','after']])
errtrain=train.loc[ytest.index[errind]]
errtrain['wrong_cls']=ytestpl[list(np.where(errind)[0])]

pd.DataFrame({'score':model.get_fscore()}).sort_values('score')

save(ytest,'xgb_result.pc')


#%% use the model
ytest = load('xgb_result.pc')
ytest.correct.mean()
0.99584612515060322


##%% model
#n=100000
#trainset=[({'a':bf},cls) for (bf,cls) in zip(train.before[:n], train.cls[:n])]
#classifier=nltk.classify.MaxentClassifier.train(trainset,"megam")
#for a,b in trainset:
#    x=classifier.prob_classify(a)
#
## sklearn
#
#Xs=sklearn.preprocessing.scale(X)
#
#n=100000
#cl=sklearn.linear_model.LogisticRegression(warm_start=True,n_jobs=-1,verbose=100,solver='sag')
#cl.fit(Xs[:n],train.cls[:n]=="DATE")
#from sklearn.model_selection import cross_val_score
#cross_val_score(cl, Xs[:n], train.cls[:n]=="DATE", cv=2)


