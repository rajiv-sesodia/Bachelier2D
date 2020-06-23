# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:36:04 2020

@author: Rajiv
"""


import numpy as np
from Bachelier import BachelierPricer
import matplotlib.pyplot as plt

def writeWeightsAndBiases(w, b, fileName):
    
    file = open(fileName,'w',newline='')

    # flattens the weights and records the dimension of each weight matrix
    weights = []
    weight_sizes = []
    for i in range(len(w)):
        weight_sizes.append(w[i].shape[0])
        weight_sizes.append(w[i].shape[1])
        weights.append(np.array(w[i]).flatten())

    flattened_weights = np.concatenate(weights)
    
    # flatten the biases
    flattened_biases = np.concatenate(b)


    # writes out the dimension and the flattened weights to a .csv file
    np.savetxt(file, [weight_sizes], delimiter=',', fmt='%i')
    np.savetxt(file, [flattened_weights], delimiter=',')
    np.savetxt(file, [flattened_biases], delimiter=',')

    file.close()


def readWeightsAndBiases(fileName):

    file = open(fileName,'r',newline='')
    
    # reads in the weights sizes and flattened weights and flattened biases
    weight_sizes = np.loadtxt(file,max_rows=1, delimiter=',',dtype=int)
    flattened_weights = np.loadtxt(file, skiprows = 0, max_rows=1, delimiter=',',dtype=float)
    flattened_biases = np.loadtxt(file, skiprows = 0, max_rows=1, delimiter=',',dtype=float)
    
    # unpack the weights into 2d arrays
    L = int(len(weight_sizes) / 2)
    c = 0
    d = 0
    weights = []
    biases = []
    for l in range(L):
        
        weights.append(np.zeros((weight_sizes[l*2], weight_sizes[l*2+1])))
        biases.append(np.zeros((weight_sizes[l*2+1])))
        
        for j in range(weight_sizes[l*2+1]):
            biases[l][j] = flattened_biases[d]
            d += 1
            
        for i in range(weight_sizes[l*2]):
            for j in range(weight_sizes[l*2+1]):
                weights[l][i][j] = flattened_weights[c]
                c += 1            
                
    file.close()
    
    return weights, biases

    
def writeDiagnostics(error_w, error_b, error_a, w, b, loss, fileName):    
    
    file = open(fileName, 'w', newline='')
    
    for i in range(1,len(error_w)):     
        np.savetxt(file, error_w[i], delimiter=',')
        
    for i in range(1,len(error_b)):         
        np.savetxt(file, error_b[i], delimiter=',')    
    
    for i in range(len(error_a)):     
        np.savetxt(file, error_a, delimiter=',')
        
    np.savetxt(file, loss, delimiter=',')
    
    for l in range(1,len(w)):
        np.savetxt(file, w[l], delimiter=',')
        
    for l in range(1,len(b)):        
        np.savetxt(file, b[l], delimiter=',')
    
    file.close()
    
    
    
def writeOutput(X_train, X_train_std, Y_train, Y_train_std, dY_train, dY_train_std,\
                X_test, X_test_std, Y_test, Y_test_std, dY_test, dY_test_std, \
                stdscX, stdscY, c, NN, eta):

    # check how well we fitted the training data 
    z_train, a_train = NN.feedForward(X_train_std)
    pv_train = stdscY.inverse_transform(a_train[NN.L-1])
    bperror = 10000*np.abs(pv_train-Y_train[:,[0]])
    dyda_train = NN.gradient(eta, X_train_std, Y_train_std, dY_train_std, c)
    delta1_train = np.zeros(X_train_std.shape[0])
    delta2_train = np.zeros(X_train_std.shape[0])
    vega1_train = np.zeros(X_train_std.shape[0])
    vega2_train = np.zeros(X_train_std.shape[0])
    rho12_train = np.zeros(X_train_std.shape[0])
    for m in range(X_train_std.shape[0]):
        delta1_train[m] = dyda_train[m][0]
        delta2_train[m] = dyda_train[m][1]
        vega1_train[m] = dyda_train[m][3]
        vega2_train[m] = dyda_train[m][4]
        rho12_train[m] = dyda_train[m][5]
    
    
    file_output = open('output_train.csv','w')
    file_output.write("F1,F2, K,vol1,vol2, rho12,T,pv_p,pv,bp_error,delta1_p,delta1,delta2_p,delta2,vega1_p,vega1,vega2_p,vega2, rho12_p, rho12 \n")
    np.savetxt(file_output, np.c_[X_train, pv_train, Y_train, bperror, \
                                  delta1_train, dY_train[:,0], delta2_train, dY_train[:,1], \
                                  vega1_train, dY_train[:,2], vega2_train, dY_train[:,3], rho12_train, dY_train[:,4]], delimiter=',')
    file_output.close()
    
    # check how well we fitted the test data 
    z_test, a_test = NN.feedForward(X_test_std)
    pv_test = stdscY.inverse_transform(a_test[NN.L-1])
    bperror = 10000*np.abs(pv_test-Y_test[:,[0]])
    dyda_test = NN.gradient(eta, X_test_std, Y_test_std, dY_test_std, c)
    delta1_test = np.zeros(X_test_std.shape[0])
    delta2_test = np.zeros(X_test_std.shape[0])
    vega1_test = np.zeros(X_test_std.shape[0])
    vega2_test = np.zeros(X_test_std.shape[0])
    rho12_test = np.zeros(X_test_std.shape[0])
    for m in range(X_test_std.shape[0]):
        delta1_test[m] = dyda_test[m][0]
        delta2_test[m] = dyda_test[m][1]
        vega1_test[m] = dyda_test[m][3]
        vega2_test[m] = dyda_test[m][4]
        rho12_test[m] = dyda_test[m][5]
    
    
    file_output = open('output_test.csv','w')
    file_output.write("F1,F2, K,vol1,vol2, rho12,T,pv_p,pv,bp_error,delta1_p,delta1,delta2_p,delta2,vega1_p,vega1,vega2_p,vega2, rho12_p, rho12 \n")
    np.savetxt(file_output, np.c_[X_test, pv_test, Y_test, bperror, \
                                  delta1_test, dY_test[:,0], delta2_test, dY_test[:,1], \
                                  vega1_test, dY_test[:,2], vega2_test, dY_test[:,3], rho12_test, dY_test[:,4]], delimiter=',')

    file_output.close()
    
    # # visual check on slices
    # F = np.arange(-0.03, 0.05,0.001)
    # V = np.linspace(0.0010, 0.0150, num = 4)
    # T = np.linspace(0.1, 5, num = 5)
    
    # fig, axis = plt.subplots(V.shape[0], T.shape[0], sharex=True, sharey=True, squeeze=True)
    # for i in range(V.shape[0]):
    #     for j in range(T.shape[0]):
            
    #         # calculation
    #         X_slice = np.array([F, np.full(F.shape[0],0.01), np.full(F.shape[0],V[i]), np.full(F.shape[0],T[j])]).T
    #         X_slice_std = stdscX.transform(X_slice)
    #         zOut, aOut = NN.feedForward(X_slice_std)
    #         pv = stdscY.inverse_transform(aOut[NN.L-1])
    #         b = BachelierPricer('call').valueSlice(X_slice)
    
    #         #plot the result  
    #         axis[i,j].tick_params(axis='both', which='major', labelsize=5)
    #         axis[i,j].plot(F,pv, linewidth=1, label = '{0:.2f},{1:.1f}'.format(V[i]*100.0,T[j]))
    #         axis[i,j].plot(F,b, linewidth=1)
    #         axis[i,j].xaxis.set_ticks(np.arange(-0.03, 0.06, 0.02))
    #         axis[i,j].yaxis.set_ticks(np.arange(0, 0.06, 0.01))
    #         axis[i,j].legend(loc='upper left', fontsize='xx-small')