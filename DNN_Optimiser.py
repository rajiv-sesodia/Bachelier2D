# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:00:05 2020

@author: Rajiv
"""
import numpy as np

class Optimiser:
    
    def __init__(self, type='SGD'):
        self.type = type
        
        
    def update(self, dcdw):
        increment  = 0
        return increment
    
    
    
class SGD(Optimiser):
    
    def __init__(self, eta = 0.1):
        self.eta = eta
        
        
    def update(self, dcdw, dcdb, l, t):
        dcdw = - self.eta * dcdw
        dcdb = - self.eta * dcdb
        return dcdw, dcdb


class Momentum(Optimiser):    
    
    def __init__(self, L, eta = 0.1, gamma = 0.9):
        self.eta = eta
        self.gamma = gamma
        self.nu_w = [0]*L
        self.nu_b = [0]*L
        
    def update(self, dcdw, dcdb, l, t):
        
        if t == 0:
            self.nu_w[l] = 0
            self.nu_b[l] = 0
        
        self.nu_w[l] = self.gamma * self.nu_w[l] + self.eta * dcdw
        self.nu_b[l] = self.gamma * self.nu_b[l] + self.eta * dcdb
        dw = -self.nu_w[l]
        db = -self.nu_b[l]
    
        return dw, db
    
    
class Adam(Optimiser):
    
    def __init__(self, L, eta = 0.1, eps = 1e-08, beta_1 = 0.9, beta_2 = 0.999):
        self.eta = eta
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_w = [0]*L
        self.v_w = [0]*L
        self.m_b = [0]*L
        self.v_b = [0]*L
 
        
    def update(self, dcdw, dcdb, l, t):
        
        if t == 0:
            self.m_w[l] = 0
            self.v_w[l] = 0
            self.m_b[l] = 0
            self.v_b[l] = 0
              
        # weights
        self.m_w[l] = self.beta_1 * self.m_w[l] + (1 - self.beta_1) * dcdw
        self.v_w[l] = self.beta_2 * self.v_w[l] + (1 - self.beta_2) * np.square(dcdw)
        m_hat_w = self.m_w[l] / (1 - np.power(self.beta_1, t+1))
        v_hat_w = self.v_w[l] / (1 - np.power(self.beta_2, t+1))
        dw = - self.eta * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
        
        # biases
        self.m_b[l] = self.beta_1 * self.m_b[l] + (1 - self.beta_1) * dcdb
        self.v_b[l] = self.beta_2 * self.v_b[l] + (1 - self.beta_2) * np.square(dcdb)
        m_hat_b = self.m_b[l] / (1 - np.power(self.beta_1, t+1))
        v_hat_b = self.v_b[l] / (1 - np.power(self.beta_2, t+1))
        db = - self.eta * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
        
        return dw, db
        
    