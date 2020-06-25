# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:52:12 2020

@author: Rajiv
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class Scaler:

    def __init__(self, name = 'MinMax'):
        self.name = name

    def Scale_MinMax(self, X_train, X_test, Y_train, Y_test, dY_train, dY_test):
            
        ScalerX = MinMaxScaler()
        X_train_ = ScalerX.fit_transform(X_train)
        X_test_ = ScalerX.transform(X_test)
        
        ScalerY = MinMaxScaler()
        Y_train_ = ScalerY.fit_transform(Y_train)
        Y_test_ = ScalerY.transform(Y_test)
        
        c = ScalerX.data_range_ / ScalerY.data_range_
        d = [c[index] for index in [0,1,3,4,5]]
        dY_train_ = dY_train * d
        dY_test_ = dY_test * d
            
        
        return X_train_, X_test_, Y_train_, Y_test_, dY_train_, dY_test_, ScalerX, ScalerY, c
            
    
    def Scale_Standard(self, X_train, X_test, Y_train, Y_test, dY_train, dY_test):
            
        ScalerX = StandardScaler()
        X_train_ = ScalerX.fit_transform(X_train)
        X_test_ = ScalerX.transform(X_test)
        
        ScalerY = StandardScaler()
        Y_train_ = ScalerY.fit_transform(Y_train)
        Y_test_ = ScalerY.transform(Y_test)
        
        c = np.sqrt(ScalerX.var_) / np.sqrt(ScalerY.var_)
        d = [c[index] for index in [0,1,3,4,5]]
        dY_train_ = dY_train * d
        dY_test_ = dY_test * d
                
        return X_train_, X_test_, Y_train_, Y_test_, dY_train_, dY_test_, ScalerX, ScalerY, c


class ActivationFunctions:
    
    def __init__(self, name = 'sigmoid', alpha = 1):
        self.alpha = alpha
        self.name = name
        self.bind()
        
    def bind(self):
        
        if self.name == 'sigmoid':
            self.phi = self.sigmoid    
            self.dphi = self.dsigmoid
        elif self.name == 'softplus':
            self.phi = self.softplus    
            self.dphi = self.dsoftplus
            self.d2phi = self.d2softplus
        elif self.name == 'linear':
            self.phi = self.linear
            self.dphi = self.dlinear
        elif self.name == 'elu':
            self.phi = self.elu
            self.dphi = self.delu
            self.d2phi = self.d2elu
        elif self.name == 'isrlu':
            self.phi = self.isrlu
            self.dphi = self.disrlu
        elif self.name == 'tanh':
            self.phi = self.tanh
            self.dphi = self.tanh
        else:
            raise RuntimeError('unknown activation function')


    def tanh(self, z): 
        return np.tanh(z)

    def dtanh(self, z):
        return 1 - np.power(np.tanh(z), 2)

    def isrlu(self, z):
        return np.where(z > 0, z, z / np.sqrt(1+self.alpha*z*z))

    def disrlu(self, z):
        return np.where(z > 0, 1, np.power( 1.0 / np.sqrt(1+self.alpha*z*z), 3.0))

    def elu(self, z):
        return np.where(z > 0, z, self.alpha*(np.exp(np.clip(z, -250, 250))-1))
        
    def delu(self, z):
        return np.where(z > 0, 1, self.elu(z) + self.alpha)

    def d2elu(self, z):
        return np.where(z > 0, 0, self.delu(z))

    def sigmoid(self, z):
        # activation function as a function of z
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))      
    
    def dsigmoid(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))    
    
    def softplus(self, z):
        return np.log(1+np.exp(np.clip(self.alpha*z, -250, 250)))      
        
    def dsoftplus(self, z):
        return self.alpha*self.sigmoid(z*self.alpha)
    
    def d2softplus(self, z):
        return self.alpha*self.alpha*self.dsigmoid(z*self.alpha)
    
    def linear(self, z):
        return z
    
    def dlinear(self, z):
        return 1.0