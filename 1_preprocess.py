#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import data and preprocess
"""

import pandas as pd
import numpy as np
import pickle
import gzip

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

stat=load('stat.pc')
train=load('train.pc')
#test=load('test.pc')
#sample=load('sample.pc')

#%% save data in pickle
train = pd.read_csv('en_train.csv')
train['class'] = train['class'].astype("category")
train.columns = ['sentence', 'token', 'cls', 'before', 'after']
train.before=train.before.astype(str,copy=False)
train.after=train.after.astype(str,copy=False)

test = pd.read_csv('en_test.csv')
test.columns = ['sentence', 'token', 'before']

sample = pd.read_csv('en_sample_submission.csv')



save(train, 'train.pc')
save(test, 'test.pc')
save(sample, 'sample.pc')


#%% make vars for cls
#classes = list(train.cls.unique())
##for c in classes:
##    train.query('cls=='+'"'+c+'"').to_csv('./c+'.csv')
#for c in classes:
#    train.eval(c+'=(cls==@c)',inplace=True)
#stat=pd.DataFrame(train[classes].sum())  # stat for all classes
#stat.columns=['n']
#stat['per']=stat.n/stat.n.sum()
#stat.sort_values('n',ascending=False,inplace=True)
#
#save(stat, 'stat.pc')
#save(train, 'train.pc')

#%% See the sentence and find rules
def sen(cls):
    sp = train.query('cls=="'+cls+'"').sample()
    sens = train.query('sentence=='+str(sp['sentence'].iloc[0]))
    bf = ' '.join(sens.before.tolist())
    af = ' '.join(sens.after.tolist())
    return bf,af,sp.before.iloc[0],sp.after.iloc[0]

def see(i,n=5):
    c=stat.index[i]
    print(c)
    print(stat.loc[c])
    for _ in range(n):
        print()
        print('\n'.join(sen(c)))
        

see(1) #PUNCT: no change at all!!!
train.query('cls=="PUNCT" & before!=after')

see(2) #DATE: 
# numbers & special words

see(3) #LETTERS
# if not a normal word
# upper case rate, end with '-'

see(4) #CARDINAL
# pure number

see(5) #VERBATIM
train.query('VERBATIM & before!=after')
# special char (only make a dict)

see(6) #MEASURE
# number + special char

see(7) # ORDINAL
# number + th/st...
# Roman number

see(8) #DECIMAL
# number+point+number

see(9) #MONEY
# $+number

see(10) #DIGIT
# number

see(11) # ELECTRONIC
# letter.com.net.,,,,

see(12) # TELEPHONE
# number-number

see(13) # TIME
# fixed 00:00:00 or am,pm

see(14) #FRACTION
# 1/2 or ⅓ like

see(15) #ADDRESS
# letter + number

digit=train.before.map(lambda x:x.isdigit())
from collections import Counter
Counter(train.loc[digit].cls).most_common()
Counter(train.loc[~digit].cls).most_common()
#%% feature engineer

# feature 1
features=['Nletter','Nupper','Nlower','Nnum','Ncomma',
          'Ndot','Nspace','Nminus','Ncolon','Nslash',
          'Nother']

def get_feature(s):
    ans=[0 for _ in range(11)]
    for c in s:
        if c.isspace():
            ans[6]+=1
        elif c=='.':
            ans[5]+=1
        elif c==',':
            ans[4]+=1
        elif c=='-':
            ans[7]+=1
        elif c==':':
            ans[8]+=1
        elif c=='/':
            ans[9]+=1
        elif c.islower():
            ans[2]+=1
        elif c.isupper():
            ans[1]+=1
        elif c.isdigit():
            ans[3]+=1
        else:
            ans[10]+=1
    ans[0]=ans[1]+ans[2]
    return ans
fdata = pd.DataFrame(train.before.map(get_feature).tolist())
fdata.columns=features

# feature 2 code the char
wlen=train.before.map(len)
wlen.quantile(0.99)
wlen.quantile(0.95)
(wlen<=10).mean() #0.9618

max_pre, max_pri=5, 5
max_both=max_pre+max_pri
def code_str(s):
    '''
    max_pre: chars from the begining
    max_pri: chars from the end
    '''
    n=len(s)
    if n<=max_pre:
        return [ord(c) for c in s]
    if n<=max_both:
        return [ord(c) for c in s[:max_pre]]+[0 for _ in range(max_both-n)]+[ord(c) for c in s[max_pre:]]
    return [ord(c) for c in s[:max_pre]] + [ord(c) for c in s[(-max_pri):]]
cdata = pd.DataFrame(train.before.map(code_str).tolist())
cdata.fillna(0,inplace=True)
cdata=cdata.astype(int,inplace=True)
cdatap1=cdata.shift(-1)
cdatam1=cdata.shift(1)
cdatap1.loc[train.sentence!=train.sentence.shift(-1)]=0
cdatam1.loc[train.sentence!=train.sentence.shift(1)]=0
cdata = pd.concat([cdata,cdatap1,cdatam1],axis=1,ignore_index=True,copy=False)
del cdatap1,cdatam1
cdata.fillna(0,inplace=True)

cdata.columns=['t'+str(i) for i in range(max_both)]+['p'+str(i) for i in range(max_both)]+['m'+str(i) for i in range(max_both)] 
# this word + latter word + previous word

X=pd.concat([fdata,cdata],axis=1,copy=False)
del fdata,cdata
for c in X.columns:
    print(X[c].max())
    if c in features:
        X[c]=X[c].astype(np.int16)
    else:
        X[c]=X[c].astype(np.int32)
save(X,'X.pc')


    
