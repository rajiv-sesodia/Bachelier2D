# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:08:18 2020

@author: Rajiv
"""


import numpy as np
import scipy.stats as stats


def N(x) : return stats.norm.cdf(x)
def n(x) : return stats.norm.pdf(x)

class BachelierPricer:
    
    def __init__(self, optionType = 'call'):
        self.optionType = optionType
        
    def value(self, F, K, vol, T):
        
        # working params
        var = self.Var(vol, T)
        d  = self.d(F, K, vol, T)
        
        # call and put prices
        call = np.sqrt(var)*(d*N(d) + n(d))
        put = np.sqrt(var)*(n(-d) - d*N(-d))
        value = (call if self.optionType == 'call' else put)

        return value
    
    def d(self, F, K, vol, T): 
        var = self.Var(vol, T)
        return (F-K)/np.sqrt(var)
        
    def Var(self, vol, T):
        return vol*vol*T
    
    
    def delta(self, F, K, vol, T):
        # working params
        var = vol*vol*T
        d  = (F-K)/np.sqrt(var)
        delta = (N(d) if self.optionType == 'call' else -N(-d))
        return delta
        
    
    def vega(self, F, K, vol, T):
        # working params
        var = vol*vol*T
        d  = (F-K)/np.sqrt(var)
        vega = np.sqrt(T)*n(d)
        return vega
    
    def valueSlice(self, X):
        result = self.value(X[:,0],X[:,1],X[:,2],X[:,3])
        return result
        
        
        
    
