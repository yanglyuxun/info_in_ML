#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:56:03 2017

@author: ylx
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
import inflect
from num2words import num2words 
import re
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
p = inflect.engine()

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object
train=load('train.pc')
stat=load('stat.pc')

#%%
stat.index
#['PLAIN', 'PUNCT', 'DATE', 'LETTERS', 'CARDINAL', 'VERBATIM', 'MEASURE',
#       'ORDINAL', 'DECIMAL', 'MONEY', 'DIGIT', 'ELECTRONIC', 'TELEPHONE',
#       'TIME', 'FRACTION', 'ADDRESS']
# 
def choose(c):
    return train.query('cls=="'+c+'"')[['before','after']]
def test_fun(c,fun):
    df=choose(c)
    df['conv']=df.before.map(fun)
    df['correct']=df.eval('after==conv')
    print(df.correct.mean())
    return df,df.query('correct==False').drop_duplicates()

#%% 'CARDINAL'
import roman

def digit2word(key):
    key=key.replace(' U.S.','').replace(',','').replace('.','').strip()
    if key.replace('0','')=='':
        return 'zero'
    if key=='-0':
        return 'minus zero'
    try:
        if key.endswith("'s"):
            return digit2word(key[:-2])+"'s"
        if key.endswith("s"):
            return digit2word(key[:-1])+"'s"
        key=str(roman.fromRoman(key.replace('IIII','IV')))
    except:
        pass
    try:
        key=str(roman.fromRoman(key.split()[0].replace('IIII','IV')))
    except:
        pass
    try:
        text = p.number_to_words(key,decimal='point',andword='', zero='o')
        if re.match(r'^0\.',key): 
            text = 'zero '+text[2:]
        if re.match(r'.*\.0$',key): text = text[:-2]+' zero'
        text = text.replace('-',' ').replace(',','')
        return text.lower()
    except: return key

df,dfw=test_fun('CARDINAL',digit2word)


#%%  'DATE'
dict_mon = {'jan': "January", "feb": "February", "mar ": "march", "apr": "april", "may": "may ","jun": "june", "jul": "july", "aug": "august","sep": "september",
            "oct": "october","nov": "november","dec": "december", "january":"January", "february":"February", "march":"march","april":"april", "may": "may", 
            "june":"june","july":"july", "august":"august", "september":"september", "october":"october", "november":"november", "december":"december"}
from dateutil import parser
def date2word(key):
    v =  key.split('-')
    if len(v)==3:
        if v[1].isdigit():
            try:
                date = datetime.strptime(key , '%Y-%m-%d')
                text = 'the '+ p.ordinal(p.number_to_words(int(v[2]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                if int(v[0])>=2000 and int(v[0]) < 2010:
                    text = text  + ' '+digit2word(v[0])
                else: 
                    text = text + ' ' + digit2word(v[0][0:2]) + ' ' + digit2word(v[0][2:])
            except:
                text = key
            return text.lower()    
    else:   
        v = re.sub(r'[^\w]', ' ', key).split()
        if v[0].isalpha():
            try:
                if len(v)==3:
                    text = dict_mon[v[0].lower()] + ' '+ p.ordinal(p.number_to_words(int(v[1]))).replace('-',' ')
                    if int(v[2])>=2000 and int(v[2]) < 2010:
                        text = text  + ' '+digit2word(v[2])
                    else: 
                        text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])   
                elif len(v)==2:

                    if int(v[1])>=2000 and int(v[1]) < 2010:
                        text = dict_mon[v[0].lower()]  + ' '+ digit2word(v[1])
                    else: 
                        if len(v[1]) <=2:
                            text = dict_mon[v[0].lower()] + ' ' + digit2word(v[1])
                        else:
                            text = dict_mon[v[0].lower()] + ' ' + digit2word(v[1][0:2]) + ' ' + digit2word(v[1][2:])
                else: text = key
            except: text = key
            return text.lower()
        else: 
            key = re.sub(r'[^\w]', ' ', key)
            v = key.split()
            try:
                date = datetime.strptime(key , '%d %b %Y')
                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                if int(v[2])>=2000 and int(v[2]) < 2010:
                    text = text  + ' '+digit2word(v[2])
                else: 
                    text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
            except:
                try:
                    date = datetime.strptime(key , '%d %B %Y')
                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                    if int(v[2])>=2000 and int(v[2]) < 2010:
                        text = text  + ' '+digit2word(v[2])
                    else: 
                        text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                except:
                    try:
                        date = datetime.strptime(key , '%d %m %Y')
                        text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                        if int(v[2])>=2000 and int(v[2]) < 2010:
                            text = text  + ' '+digit2word(v[2])
                        else: 
                            text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                    except:
                        try:
                            date = datetime.strptime(key , '%d %m %y')
                            text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                            v[2] = datetime.date(date).strftime('%Y')
                            if int(v[2])>=2000 and int(v[2]) < 2010:
                                text = text  + ' '+digit2word(v[2])
                            else: 
                                text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                        except:text = key
            return text.lower() 
def date2word2(s):
    if len(s)==4 and s.isdigit():
        if int(s)>=2000 and int(s) < 2010:
            return digit2word(s)
        else:
            return digit2word(s[0:2]) + ' ' + digit2word(s[2:])
    return date2word(s)
df,dfw=test_fun('DATE',date2word2)