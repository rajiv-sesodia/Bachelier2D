# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:52:12 2020

@author: Rajiv
"""

import numpy as np

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