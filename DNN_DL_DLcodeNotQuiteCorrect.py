# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""

import numpy as np
from ActivationFunctions import ActivationFunctions
from DNN_Helper import readWeightsAndBiases, writeWeightsAndBiases, writeDiagnostics

# One of my first Neural Networks
class NeuralNetwork:
    
    def __init__(self, N, L2 = 0, alpha = 0.01, beta = 1.0):
        # L is the number of layers in the neural network
        # N is an array containing the number of nodes in each layer
        self.L = N.shape[0]
        self.N = N
        self.af = ActivationFunctions('elu', alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
        self.d2phi = self.af.d2phi
        self.L2 = L2
        self.beta = beta

        
    def initialise(self, weightsAndBiasesFile=''):
    
        # initialise weights and biases
        self.w = []
        self.b = []
        
        # if weights file supplied, read them in
        if weightsAndBiasesFile:
            self.w, self.b = readWeightsAndBiases(weightsAndBiasesFile)
            return
            
        # otherwise set randome weights
        # always important to set the seed for comparability
        np.random.seed(0)
        
        # these are members of the NN class
        self.w.append(np.zeros((self.N[0],self.N[0])))
        self.b.append(np.zeros(self.N[0]))
        
        # the initial weights and biases are set to a random number taken from a (standard) normal distribution, i.e.
        # with mean 0 and variance 1
        for l in range(1,self.L):
            # self.w.append(np.random.normal(0.0,1.0,(self.N[l-1],self.N[l])))
            # self.b.append(np.random.normal(0.0,1.0,self.N[l]))
            self.w.append(np.sqrt(2.0/self.N[l])*np.random.rand(self.N[l-1],self.N[l]))
            self.b.append(np.sqrt(2.0/self.N[l])*np.random.rand(self.N[l]))
            
               
    def feedForward(self, X):                
        
        # calculates the perceptron value (z) and activated value (a) through the activation function
        X = X if X.ndim > 1 else X.reshape(1,X.shape[0])
        
        # note, no activation for input layer, a = X
        a = [X]
        z = [np.zeros((X.shape[0],X.shape[1]))]
        for l in range(0,self.L-1):            
            z.append(a[l].dot(self.w[l+1]) + self.b[l+1])
            a.append(self.phi(z[l+1]))
            
        return z, a
    
    
    def calc_dcdz(self, a, z, y, dy):
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        
        # special treatment for last layer as this derivative is from the cost function itself.
        # for the remainder of the layers following recursion formulae        # dcdz = [(2.0 / self.N[self.L-1]) * np.multiply( (a[self.L-1]-y) , self.dphi(z[self.L-1])) ]
        m = a[self.L-1].shape[0]
        dcdz = [(2.0 / m) * np.multiply( (a[self.L-1]-y) , self.dphi(z[self.L-1])) ]
                
        dyda = [np.ones((m, self.N[self.L-1]))]
        for l in reversed(range(0,self.L-1)):
            dcdz.insert(0, np.multiply( dcdz[0].dot((self.w[l+1]).T), self.dphi(z[l])))
            dyda.insert(0, np.multiply(dyda[0], self.dphi(z[l+1])).dot(self.w[l+1].T))
        
        # derivative of z^L w.r.t. a^(L-1)
        dzda = self.w[self.L-1].T
        for l in reversed(range(0,self.L-2)):
            dzda = np.multiply(dzda, self.dphi(z[l+1])).dot(self.w[l+1].T)
        
        # rescale and calculate derivative of (derivative) cost function - only look at index 0 and 2 (delta and vega)
        da0bar_dzL = dzda * (self.d2phi(z[self.L-1]) ) + self.dphi(z[self.L-1])
        # to start - delta
        # part1 = np.sqrt(dy[:,0].reshape(-1,1) * dy[:,0].reshape(-1,1))
        dcbardz = [(2.0 / m) * (dyda[0][:,0]-dy[:,0]).reshape(-1,1) * da0bar_dzL[:,0].reshape(-1,1)]
        for l in reversed(range(0,self.L-1)):
            dcbardz.insert(0,np.multiply( dcbardz[0].dot((self.w[l+1]).T), self.dphi(z[l])))
    
        # add the delta penalty
        for l in reversed(range(0,self.L)):
            dcdz[l] = (1 - self.beta)* dcdz[l] + self.beta * dcbardz[l]
        
            
        return dcdz, dyda

    # backpropogation of cost calculated at output layer through the network, updating the weights and biases as we go along
    # returns dcdz which is needed for derivative
    def backProp(self, optimiser, z, a, y, dy, t):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz, dyda = self.calc_dcdz(a, z, y, dy)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        for l in reversed(range(1,self.L)):
            dcdw = a[l-1].T.dot(dcdz[l]) + self.L2*2.0*self.w[l]
            dcdb = np.sum(dcdz[l], axis = 0)
            
            # increments to the weights and biases
            dw, db = optimiser.update(dcdw, dcdb, l, t)
            self.w[l] += dw
            self.b[l] += db
        
    
    def calcLoss(self, X, Y, dY):
        m = Y.shape[0]
        z, a = self.feedForward(X)
        dcdz, dyda = self.calc_dcdz(a, z, Y, dY)
        
        # loss on values
        loss1 = (1-self.beta)*np.sum(np.square(Y-a[self.L-1])) / m
        
        # loss on gradient - first variable only (delta)
        loss2 = self.beta * np.sum(np.square(dY[:,0]-dyda[0][:,0])) / m
        
        # loss on bias
        loss3 = 0
        for l in range(self.L):
            loss3 += self.L2* np.sum(np.square(self.w[l])) / m

        return loss1, loss2, loss3
    
     
    def fit(self, optimiser, L2, epochs, X, Y, dY, batchSize, loss, weightsAndBiasesFile='', diagnosticsFile=''):
        rgen = np.random.RandomState(1)
        
        for epoch in range(epochs):
            
            # shuffle
            r = rgen.permutation(len(Y))

            # loop over entire randomised set
            for n in range(0, len(Y) - batchSize + 1, batchSize):
                indices = r[n : n + batchSize]
                z, a = self.feedForward(X[indices])
                self.backProp(optimiser, z, a, Y[indices], dY[indices], epoch)                
        
            # check error
            if epoch % 10 == 0:
                loss1, loss2, loss3 = self.calcLoss(X,Y,dY)
                print('epoch = ', epoch, 'loss1 = ',loss1, 'loss2 = ', loss2, 'loss3 = ', loss3, 'eta = ', optimiser.eta)
                loss.append([epoch, loss1 + loss2])

        if weightsAndBiasesFile:
            writeWeightsAndBiases(self.w, self.b, weightsAndBiasesFile)
            
        if diagnosticsFile:
            error_w, error_b = self.GradientCheck(X,Y,dY)
            error_a = self.feedForwardCheck(X[0])
            writeDiagnostics(error_w, error_b, error_a, self.w, self.b, loss, diagnosticsFile)
            
            

          
                
    def gradient(self, eta, X, Y, dY, stdscX, stdscY):
        
        # derivative calculation
        z, a = self.feedForward(X)
        dcdz, dyda = self.calc_dcdz(a, z, Y, dY)   
        # derivatives = np.multiply(dyda[0], stdscY.data_range_ / stdscX.data_range_)
       
        # re-scale the output
        derivatives = np.divide(dyda[0], np.sqrt(stdscX.var_))
        if stdscY.with_std:
            derivatives = np.multiply(derivatives, np.sqrt(stdscY.var_))
    
        
        return derivatives
    
    

    def backPropGradientCheck(self, eta, z, a, y, dy):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz, dyda = self.calc_dcdz(a, z, y, dy)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        dcdw = [a[self.L-2].T.dot(dcdz[self.L-1]) + self.L2*2.0*self.w[self.L-1]]
        dcdb = [np.sum(dcdz[self.L-1], axis = 0)]
        for l in reversed(range(1,self.L-1)):
            dcdw.insert(0, a[l-1].T.dot(dcdz[l]) + self.L2*2.0*self.w[l])
            dcdb.insert(0, np.sum(dcdz[l], axis = 0))
        
        dcdw.insert(0,0)
        dcdb.insert(0,0)
        
        return dcdw, dcdb
    


    def GradientCheck(self, X, Y, dY):        
        
        # base case
        eps = 1e-05
        if self.w == []:
            self.initialise()
        
        # calculate derivative
        z, a = self.feedForward(X)
        dcdw, dcdb = self.backPropGradientCheck(1.0, z, a, Y, dY)
        
        # calculate error in weights
        for n in range(1, self.L):
            for lm1 in range(self.N[n-1]):
                for l in range(self.N[n]):
                    
                    # up
                    self.w[n][lm1][l] += eps
                    C_up1, C_up2, C_up3 = self.calcLoss(X, Y, dY)
                    
                    # down
                    self.w[n][lm1][l] -= 2.0*eps
                    C_dn1, C_dn2, C_dn3 = self.calcLoss(X, Y, dY)
                    
                    # error in deriv
                    dcdw[n][lm1][l] -= (C_up1 + C_up2 - C_dn1 - C_dn2) / (2.0 * eps)
                    
                    # restore original value
                    self.w[n][lm1][l] += eps
                                      
     

        # calculate error in bias
        for n in range(1, self.L):    
            for l in range(self.N[n]):
                
                # up
                self.b[n][l] += eps
                C_up1, C_up2, C_up3 = self.calcLoss(X, Y, dY)
                
                # down
                self.b[n][l] -= 2.0 * eps
                C_dn1, C_dn2, C_dn3 = self.calcLoss(X, Y, dY)
                
                # error in deriv
                dcdb[n][l] -= (C_up1 + C_up2 - C_dn1 - C_dn2) / (2.0 * eps)
                
                # restore original value
                self.b[n][l] += eps
            
        return dcdw, dcdb
            
        
    def feedForwardCheck(self, X):         
        
        # temporary override of class members
        import copy
        w_ = copy.deepcopy(self.w)
        b_ = copy.deepcopy(self.b)
        name = self.af.name
        alpha = self.af.alpha
        self.af = ActivationFunctions('linear')
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
            
        for l in range(1,self.L):
            for i in range(self.N[l-1]):
                for j in range(self.N[l]):
                    self.w[l][i][j] = 1.0/(self.N[l-1])
                   
        for l in range(1,self.L):        
            for j in range(self.N[l]):
                self.b[l][j] = 0.0
            
        result = np.full(self.N[self.L-1], np.average(X))
        

        z, a = self.feedForward(X)
        error = result - a[self.L-1]
        
        # restore class members to original values
        self.w = copy.deepcopy(w_)
        self.b = copy.deepcopy(b_)
        self.af = ActivationFunctions(name, alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
        
        return error
        