# -*- coding: utf-8 -*-
"""

helper functions

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import gym
from gym import spaces
import os, sys, operator
import imp
import multiprocessing
from contextlib import contextmanager
from copy import deepcopy
import scipy as sc
import json
import copy

def catch(func, handle=lambda e : e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)
    
def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
 
def get_keys(data, param): 
    if param not in data.keys():
        mask = [hasattr(data[key],'iteritems') for key in data.keys()]
        main_key = [v for v in np.array(data.keys())[mask] if type(catch(lambda : data[v][param]))!=KeyError]
        keys = [main_key[0], param]
    else:
        keys = param
    return keys

def check_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def run_as_process(func, *args):
    p = multiprocessing.Process(target=func, args=args)
    p.daemon = True
    try:
        p.start()
        p.join()
    finally:
        p.terminate()

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def add(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

def JS(p, q, eps=np.e):
    '''
    Jensen-Shannon divergence between given p,q vectors
    '''
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    
    js =  sc.stats.entropy(p,m, base=eps)/2. + sc.stats.entropy(q, m, base=eps)/2.
    return js

def KL(p,q,eps=0.00001):
    """
    Kullback-Leibler divergence between given p,q vectors
    """
    p = np.asarray(p)+eps
    q = np.asarray(q)+eps
    kl = np.sum(p*np.log(p/q))
     
    return kl

def find_nearest(array, value):
    """
    find closes element idx to value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def part(n, k):
    """
    create  combinations of integer n to k bins
    """
    def memoize(f):
        cache = [[[None] * n for j in xrange(k)] for i in xrange(n)]
        def wrapper(n, k, pre):
            if cache[n-1][k-1][pre-1] is None:
                cache[n-1][k-1][pre-1] = f(n, k, pre)
            return cache[n-1][k-1][pre-1]
        return wrapper

    @memoize
    def _part(n, k, pre):
        if n <= 0:
            return []
        if k == 1:
            if n <= pre:
                return [(n,)]
            return []
        ret = []
        for i in xrange(min(pre, n), 0, -1):
            ret += [(i,) + sub for sub in _part(n-i, k-1, i)]
        return ret
    return _part(n, k, n)


def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p


class Database():
    #following @tmoer
    ''' Database '''
    def __init__(self,max_size,batch_size):
        self.max_size = max_size        
        self.batch_size = batch_size
        self.size = 0
        self.insert_index = 0
        self.experience = []
        self.sample_array = None
        self.sample_index = 0
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        
    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0
                            
    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle() # Reset for the next epoch
            raise(StopIteration)
          
        if (self.sample_index + 2*self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index+self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size
        
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add) 
        return tuple(arrays)
            
    next = __next__

def stable_normalizer(x,temp):
    #following @tmoer
    x = x / np.max(x)
    return (x ** temp)/np.sum(x ** temp)

def normalizer(x,temp):
    #following @tmoer
    return np.abs((x ** temp)/np.sum(x ** temp))

def check_space(space):   
    #following @tmoer
    '''check the properties of the env '''
    if isinstance(space,spaces.Box):
        dim = space.shape # should the zero be here?
        discrete = False    
    elif isinstance(space,spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError
    return dim, discrete

def rollout(s,Env,sess,policy,model,gamma,roll_max=250):
    #following @tmoer
    ''' Small rollout function to estimate V(s)
    policy = random or targeted'''
    terminal = False
    R = 0.0
    for i in range(roll_max):
        if policy == 'random':
            a = Env.action_space.sample()
        elif policy == 'targeted':
            pi = np.squeeze(model.predict_pi(s[None,],sess))
            a = np.random.choice(len(pi),p=pi)
        s1,r,terminal,_ = Env.step(a)
        R += (gamma**i)*r
        s = s1
        if terminal:
            break
    return R
    
