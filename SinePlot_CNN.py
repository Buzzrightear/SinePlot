#Partly taken from https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/ 
#Sine wave prediction using CNN
import math
import matplotlib.pyplot as plt
#import torch
#import torch.nn as nn
import numpy as np
import pandas as pd

# univariate data preparation
from numpy import array


#CNN requires data preprocessing
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
raw_seq_SeedValue = 0.083

# define input sequence
raw_seq = []
testSet = []

# Create array of values for sine wave
for i in range(1000):
    raw_seq.append(math.sin(raw_seq_SeedValue))
    raw_seq_SeedValue += 0.1

for i in range(100):
    testSet.append(math.sin(raw_seq_SeedValue))
    raw_seq_SeedValue += 0.1

 

# choose a number of time steps
n_steps = 10

#Number of features - 1 because with univariate sequence, we just have the one variable (I think this relates to efectively having a single value input)
n_features = 1

# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
#for i in range(len(X)):
#	print('Sample number ' + str(i) + ':', X[i], y[i])


# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse') 
#'The model is fit using the efficient Adam version of stochastic gradient descent and optimized using the mean squared error, or mse, loss function.'
#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ 

# reshape from [samples, timesteps] into [samples, timesteps, features] - we have multiple samples, so input data needs to include number of variables/features being supplied/expected
X = X.reshape((X.shape[0], X.shape[1], n_features))

# We have created the structure of the model so now we have to fit the model to our training dataset
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array(testSet)
x_input = x_input.reshape((1, n_steps, n_features)) #Suggests a single sample of 10 steps?
yhat = model.predict(x_input, verbose=0)