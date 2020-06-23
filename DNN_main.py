import numpy as np
import pandas as pd
from ActivationFunctions import Scaler
from sklearn.model_selection import train_test_split
from DNN_DL import NeuralNetwork
from DNN_Helper import writeOutput
from DNN_Optimiser import SGD, Adam, Momentum


# data - you;ll have to change this path of course depending on where you put your data
df = pd.read_excel("C:/Users/Rajiv/Google Drive/Science/Python/sandbox/DNN/Regression/Bachelier2D/Generator2D.xlsx",usecols="S:AE")

#extract the training sample
X, Y, dY = df.iloc[:, [0,1,2,3,4,5,6]].values, df.iloc[:, [7]].values, df.iloc[:, [8,9,10,11,12]].values

# split into training batch and test batch
# the training batch is used to train the neural network and the test batch is used to test the trained network
X_train, X_test, Y_train, Y_test, dY_train, dY_test = train_test_split(X, Y, dY, test_size=0.2, shuffle = False, random_state=0)

# scaling for X
s = Scaler('MinMax')
X_train_, X_test_, Y_train_, Y_test_, dY_train_, dY_test_, scalerX, scalerY, c = s.Scale_MinMax(X_train, X_test, Y_train, Y_test, dY_train, dY_test)

# Now create the basic structure of the Neural Network, essentially the number of nodes at each layer. Size of N is the number of layers
N = np.array([7,50,50,1]) #number of nodes in each layer

# basic error checking on inputs. Should have more checks here I guess
if N[N.shape[0]-1] != Y_train.shape[1]:
    raise RuntimeError('Last layer must be equal to the number of class variables')
    
if X.shape[1] != N[0]:
    raise RuntimeError('First Layer must be equal to the number of inputs')
    
# hyperparameters
eta = 0.1
L2 = 0
alpha = 1.0
NN = NeuralNetwork(N, L2, alpha)
NN.initialise('')

# loss is a vector showing how the loss varies with each iteration of the algorithm (epoch)
loss = []

# we do the calculation in batches as it is more efficient
batchSize = 50
epochs = 200

# optimiser - figure out how to do this nicely later
optimiserType = 'Adam'
optimiser = 0
if optimiserType == 'SGD':
    optimiser = SGD(eta)
elif optimiserType == 'Adam':
    optimiser = Adam(NN.L, eta, 1e-08, 0.9, 0.999)
elif optimiserType == 'Momentum':
    optimiser = Momentum(NN.L, eta, 0.9)
else:
    raise RuntimeError('Incorrect choice of optimiser')

# fit the data
NN.fit(optimiser, L2, epochs, X_train_, Y_train_, dY_train_, batchSize, loss, '','')

# write the output to file
writeOutput(X_train, X_train_, Y_train, Y_train_, dY_train, dY_train_, \
            X_test, X_test_, Y_test, Y_test_, dY_test, dY_test_, \
            scalerX, scalerY, c, NN, eta)



    













