#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxEnt
"""
import numpy as np
import pandas as pd
#import math
#from sklearn.feature_extraction import DictVectorizer
from sparse_class import sparse
#from functools import reduce
from nltk.probability import DictionaryProbDist

class my_MaxEnt():
    'A maximum entropy classifier'
    def __init__(self,fset,label):
        enc, self.n_f, self.encf,self.labels=self.sparse_data(fset, label)
        self.lambdas = np.zeros(self.n_f)
    def train(self, fset, label, max_iter=30, th=1e-5,
              max_iter2=300,th2=1e-12):
        '''
        Train the MaxEnt model by a feature set fset
        Use Improved Iterative Scaling to solve the model
        :fset: a list of samples, each sample is a dict that contains
            the features
            e.g. [{'f1':0,'f2':1},{'f1':1,'f3':5}]
        :label: a list of labels. There must be len(fset)==len(label)
            e.g. ['l1','l2']
        '''
        # store the data in a sparce way so that
        # it is easier to retrieve a value
        enc, self.n_f, self.encf,self.labels=self.sparse_data(fset, label)
#        # make a aparce matrix of fset
#        fset2=({i:str(d[i]) for i in d} for d in fset)
#        v=DictVectorizer(np.bool)
#        X = v.fit_transform(fset2)
        # feature freq
        ff = np.zeros(self.n_f)
        for f, l in zip(fset,label):
            for i, v in self.encf(f, l):
                ff[i] += v
        ff=ff/len(fset) # normalize
        
        # map nf to integers from 0
        mapnf = set()
        for f in fset:
            for l in self.labels:
                mapnf.add(sum(v for (id,v) in self.encf(f, l)))
        mapnf = dict((nf, i) for (i, nf) in enumerate(mapnf))
        mapnp = np.array(sorted(mapnf, key=mapnf.get),np.float)
        mapnpt = mapnp.reshape((-1,1))
        ## use IIS
        lambdas=self.lambdas
        for _a in range(max_iter):
            print('iter:',_a)
            # solve the equation by Newton method
            deltas = np.ones(self.n_f)
            # compute a matrix for sum(p(x)p(y|x)f(x,y))
            pre = np.zeros((len(mapnf),self.n_f),np.float)
            for f in fset:
                for l in self.labels:
                    fvec=self.encf(f,l)
                    nf=np.sum(v for (id,v) in fvec) 
                    for id,v in fvec:
                        pre[mapnf[nf],id] += self.pred_prob(f,lambdas).prob(l) * v
            pre = pre/len(fset)
            # use Newton to solve the equation
            for _b in range(max_iter2):
                outer_mlp = np.outer(mapnp,deltas)
                exp_item = 2 ** outer_mlp
                t_exp = mapnpt * exp_item
                s1 = np.sum(exp_item * pre, axis=0)
                s2 = np.sum(t_exp * pre, axis=0)
                s2[s2==0]=1 #avoid 0
                dd = (ff-s1) / -s2
                deltas -= dd
                if np.abs(dd).sum()<th2 * np.abs(deltas).sum():
                    break
            lambdas += deltas
            self.lambdas=lambdas
            rate = np.abs(deltas).sum()/np.abs(lambdas).sum()
            print('R_lambda:',rate)
            if rate<th:
                break
        self.lambdas=lambdas
        return self
    
    def sparse_data(self,fset, label):
        # use the feature encoding class in nltk to store the sparse
        # matrix
        enc = sparse.store([n for n in zip(fset,label)])
        return enc,enc.length(),enc.encode,enc.labels()
        
    def pred_prob(self,f,w=None):
        'predict the probs of the labels'
        ps={}
        if w is None:
            w=self.lambdas
        for l in self.labels:
            fcode=self.encf(f,l)
            t=0.0
            for id,v in fcode:
                t+=w[id]*v
                ps[l]=t
        dist=DictionaryProbDist(ps,True,True)
        return dist
    def pred_prob_many(self,fset,l,w=None):
        ans=[]
        for f in fset:
            ans.append(self.pred_prob(f,w).prob(l))
        return ans
    def pred_class(self,f,w=None):
        ps = self.pred_prob(f,w)
        return ps.max()
    def pred_class_many(self,fset,w=None):
        ans=[]
        for f in fset:
            ans.append(self.pred_class(f,w))
        return ans
    
    def save(self,fname='my_maxent.model'):
        pd.io.pickle.to_pickle(self,fname,'gzip')
    def load(self,fname='my_maxent.model'):
        self = pd.io.pickle.read_pickle(fname,'gzip')
        return self


