#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
info measure
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
import IB.IB as ib
from maxent import my_MaxEnt

save = lambda o,f:pd.io.pickle.to_pickle(o,f,'gzip')
load = lambda f:pd.io.pickle.read_pickle(f,'gzip')

stat=load('stat.pc')
#train=load('train.pc')
#trains,tests=load('train_test.pc')
X=load('smallX.pc')

#%% sample small data
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
# stat the accuracy for each cat
#cl_maxent=pd.io.pickle.read_pickle('max_ent_model.pc','gzip')
#ans=[]
#for cat in stat.index:
#    testnew=[s for s in tests if s[1]==cat]
#    if len(testnew)==0: continue
#    acc=nltk.classify.accuracy(cl_maxent.pipclassifier, testnew)
#    ans.append([cat,len(testnew),acc])

#%% small data set
X = train.sample(1000,random_state=777)
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
X['feature']=[features(row) for row in X.as_matrix()]
convert2=lambda x: True if x=='PLAIN' else False
X['isplain']=X.cls.map(convert2)
pd.io.pickle.to_pickle(X, 'smallX.pc','gzip')

#%% measure entropy
combine = lambda row: ''.join(row[['tm1','before','tp1']].tolist())
#X = X.sample(len(X),random_state=777)#
#ntrain = int(len(X)*0.6)#
Xtxt = X.apply(combine,axis=1)
tf2txt=lambda x:str(int(x))
ytxt = X.isplain.map(tf2txt).tolist()
#Xtxt,Xtxtt = Xtxt[:ntrain],Xtxt[ntrain:]#
#ytxt,ytxtt = ytxt[:ntrain],ytxt[ntrain:]#

fset = [i[0] for i in X.feature.tolist()]#[:ntrain]]
#fsett = [i[0] for i in X.feature.tolist()[ntrain:]]
yset = X.isplain.tolist()#[:ntrain]
#ysett = X.isplain.tolist()[ntrain:]

def nx(x):
    c=Counter(x)
    return np.array(list(c.values()))
def ent(x):
    n=nx(x)
    p=n/n.sum()
    return ib.entropy(p)
def cxy(x,y):
    return Counter([(a,b) for a,b in zip(x,y)])
def Hxy(x,y):
    nxy = np.array(list(cxy(x,y).values()))
    return ib.entropy(nxy/nxy.sum())
def Ixy(x,y,Hx=None):
    if Hx is None: # Hx is the precomputed value
        return ent(x)+ent(y)-Hxy(x,y)
    else:
        return Hx+ent(y)-Hxy(x,y)

Hx = ent(Xtxt)
Hy = ent(ytxt)
ent(Xtxt)
9.8985650037132142
ent(ytxt)
0.81599847897086653
Hxy(Xtxt,ytxt)
9.8985650037132142
Ixy(Xtxt,ytxt)
0.81599847897086697
    
#%% find the informational bottleneck
def pxy(x,y):
    c=cxy(x,y)
    xval = set()
    yval =set()
    for xv,yv in c:
        xval.update(xv)
        yval.update(yv)
    tb = pd.DataFrame(0,index=list(xval),columns=list(yval))
    for xv,yv in c:
        tb.loc[xv,yv]=c[(xv,yv)]
    tb=tb.fillna(0)
    tb = tb.as_matrix()
    return tb/tb.sum()
ds_ib = ib.dataset(pxy(Xtxt,ytxt))
fit_param = pd.DataFrame(data={'betas':[1,1.1,1.3,1.5,1.8]})
fit_param['alpha'] = 1
fit_param['repeats'] = 1
fit_param['max_time'] = 30*60 # 30min
metrics_conv, dist_conv, metrics_sw, dist_sw = ib.IB(ds_ib,fit_param)
save([metrics_conv, dist_conv, metrics_sw, dist_sw],'ib_result.pc')

# analyze
metrics_conv, dist_conv, metrics_sw, dist_sw = load('ib_result.pc')
ibdata = metrics_conv.query('clamped==True')
ibxt,ibyt = ibdata.ixt, ibdata.iyt
plt.plot(ibxt,ibyt)
#%% train the Max Ent model and find the plot
cl=my_MaxEnt(fset,yset)
ttxt = [str(round(x,1)) for x in cl.pred_prob_many(fset,True)]
#ttxtt = [str(round(x,1)) for x in cl.pred_prob_many(fsett,True)]
Ixt = [Ixy(Xtxt,ttxt,Hx)]
Iyt = [Ixy(ytxt,ttxt,Hy)]
#Ixtt = [Ixy(Xtxtt,ttxtt)]
#Iytt = [Ixy(ytxtt,ttxtt)]
for it in range(30):
    cl.train(fset, yset, max_iter=1)
    ttxt = [str(round(x,1))  for x in cl.pred_prob_many(fset,True)]
#    ttxtt = [str(round(x,1))  for x in cl.pred_prob_many(fsett,True)]
    Ixt.append(Ixy(Xtxt,ttxt,Hx))
    Iyt.append(Ixy(ytxt,ttxt,Hy))
#    Ixtt.append(Ixy(Xtxtt,ttxtt))
#    Iytt.append(Ixy(ytxtt,ttxtt))
    print('real_it:',it)
p0=plt.figure()
plt.plot(Ixt,Iyt,'.-')
plt.xlabel('I(X;T)')
plt.ylabel('I(Y;T)')
plt.title('The learning line of MatEnt model')
save(p0,'plot_maxent.pc')


#%% MLP NN classifier

def get_net(cl,X):
    '''modified from sklearn.neural_network.MLPClassifier._predict()'''
#    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    # Make sure self.hidden_layer_sizes is a list
    hidden_layer_sizes = cl.hidden_layer_sizes
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [cl.n_outputs_]
    # Initialize layers
    activations = [X]
    for i in range(cl.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],
                                     layer_units[i + 1])))
    # forward propagate
    cl._forward_pass(activations)
    return activations[1:]
def calc_info(net,Xtxt,ytxt):
    Ixt = []
    Iyt = []
    for mat in net:
        Ixt1 = []
        Iyt1 = []
        for j in range(mat.shape[1]):
            ttxt = list(mat[:,j].round(1).astype(str))
            Ixt1.append(Ixy(Xtxt,ttxt,Hx))
            Iyt1.append(Ixy(ytxt,ttxt,Hy))
        Ixt.append(np.max(Ixt1))
        Iyt.append(np.max(Iyt1))
    return Ixt,Iyt

nk = 5
from sklearn.neural_network import MLPClassifier

#sol = 'sgd'
#sol = 'adam'
sol = 'lbfgs'
skcl = [MLPClassifier(hidden_layer_sizes=(20,15,10),
                    max_iter=1,
                    warm_start=True,
                    solver=sol, # default: 'adam'
                    activation='tanh') for k in range(nk)]
nncl = [nltk.classify.scikitlearn.SklearnClassifier(k) for k in skcl]
Xtrans = nncl[0]._vectorizer.fit_transform(fset)
ytrans = nncl[0]._encoder.fit_transform(yset)
Ixt = [[] for k in range(nk)]
Iyt = [[] for k in range(nk)]
for i in range(100):
    for k in range(nk):
        nncl[k].train(list(zip(fset,yset)))
        #Xtranst = nncl[k]._vectorizer.transform(fsett)
        #ytranst = nncl[k]._encoder.transform(ysett)
        net = get_net(nncl[k]._clf,Xtrans) ##
        Ixt0,Iyt0 = calc_info(net,Xtxt,ytxt) ##
        Ixt[k].append(Ixt0)
        Iyt[k].append(Iyt0)
    print(i)
def avg_mats(I):
    I0 = [np.array(i) for i in I]
    while len(I0)>1:
        I0 = [I0[0]+I0[1]]+I0[2:]
    return I0[0]/len(I)
p=plt.figure()
for i in range(len(Ixt[0][0])):
    xxx=[n[i] for n in avg_mats(Ixt)]
    yyy=[n[i] for n in avg_mats(Iyt)]
    #plt.plot(xxx,yyy,alpha=0.3)
    plt.plot(xxx,yyy,'x:',alpha=0.3,label='Layer '+str(i+1))
    plt.annotate(str(i+1),(xxx[0],yyy[0]))
plt.legend()
plt.xlabel('I(X;T)')
plt.ylabel('I(Y;T)')
if sol == 'sgd':
    plt.title('The learning line of SGD - MLPNN model')
    save(p,'plot_NN_sgd.pc')   
if sol == 'adam':
    plt.title('The learning line of ADAM - MLPNN model')
    save(p,'plot_NN_adam.pc')   
if sol == 'lbfgs':
    plt.title('The learning line of L-BFGS - MLPNN model')
    save(p,'plot_NN_lbfgs.pc')   


#%% Adaboost
from sklearn.ensemble import AdaBoostClassifier
Ixt = [0]
Iyt = [0]
for n_est in range(50):
    Ixt0,Iyt0 = [],[]
    for i in range(10): # make more random trials and take avg
        skcl = AdaBoostClassifier(n_estimators=n_est+1, random_state=(n_est+i*777))
        nncl = nltk.classify.scikitlearn.SklearnClassifier(skcl)
        nncl.train(list(zip(fset,yset)))
        ttxt = [str(round(x.prob(True),1)) for x in nncl.prob_classify_many(fset)]
        Ixt0.append(Ixy(Xtxt,ttxt,Hx))
        Iyt0.append(Ixy(ytxt,ttxt,Hy))
    Ixt.append(np.mean(Ixt0))
    Iyt.append(np.mean(Iyt0))
    print(n_est)
p1=plt.figure()
plt.plot(Ixt,Iyt,'.-', alpha=0.5)
plt.xlabel('I(X;T)')
plt.ylabel('I(Y;T)')
plt.title('The learning line of AdaBoost model')
save(p1,'plot_adaboost.pc')


#%% feature entropy
def dic2txt(d):
    d=sorted(d.items())
    t=''
    for i in d:
        t+= str(i[0])+str(i[1])
    return t
def small_features(row):
    f={}
    for i,c in enumerate(row[1]):
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
    return dic2txt(f)
def tiny_features(row):
    f={}
    for i,c in enumerate(row[1]):
        if c.isspace():
            l='space'
            f[l]=1
        elif c=='.':
            l='dot'
            f[l]=1
        elif c==',':
            l='comma'
            f[l]=1
        elif c=='-':
            l='minus'
            f[l]=1
        elif c==':':
            l='twodot'
            f[l]=1
        elif c=='/':
            l='slash'
            f[l]=1
        elif c.islower():
            l='low'
            f[l]=1
        elif c.isupper():
            l='up'
            f[l]=1
        elif c.isdigit():
            l='dig'
            f[l]=1
        else:
            l='other'
            f[l]=1
    return dic2txt(f)
ent([small_features(row) for row in X.as_matrix()])
4.6664539086993688/9.8985650037132142
Ixy([small_features(row) for row in X.as_matrix()],ytxt)
0.77664007108568178/0.81599847897086653
ent([tiny_features(row) for row in X.as_matrix()])
2.2262494835047613/9.8985650037132142
Ixy([tiny_features(row) for row in X.as_matrix()],ytxt)
0.75927057975226253/0.81599847897086653
ent(Xtxt)
9.8985650037132142
ent(ytxt)
0.81599847897086653
