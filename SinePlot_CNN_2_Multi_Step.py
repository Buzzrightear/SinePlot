#Partly taken from https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/ 
#Sine wave prediction using CNN
import numpy as np
import PyGnuplot as pg
import math
import matplotlib.pyplot as plt
#import torch
#import torch.nn as nn
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# univariate data preparation
from numpy import array
 
# split a univariate sequence into samples for multistep prediction
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y) 
 
 #Sort input sequence:
raw_seq_SeedValue = 0.083

# define input sequence
raw_seq = []
testSet = []

# choose a number of time steps for each sample
n_steps_in = 30
n_steps_out = 20

# Create array of values for sine wave
for i in range(1000):
    raw_seq.append(math.sin(raw_seq_SeedValue))
    raw_seq_SeedValue += 0.1

raw_seq_SeedValue = 0.013
for i in range(n_steps_in):
    testSet.append(math.sin(raw_seq_SeedValue))
    raw_seq_SeedValue += 0.1

# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

#Number of features - 1 because with univariate sequence, we just have the one variable (I think this relates to efectively having a single value input)
n_features = 1

# reshape from [samples, timesteps] into [samples, timesteps, features] - we have multiple samples, so input data needs to include number of variables/features being supplied/expected
X = X.reshape((X.shape[0], X.shape[1], n_features))


# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse') 
#'The model is fit using the efficient Adam version of stochastic gradient descent and optimized using the mean squared error, or mse, loss function.'
#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ 
# We have created the structure of the model so now we have to fit the model to our training dataset
model.fit(X, y, epochs=2000, verbose=0)

# demonstrate prediction
x_input = array(testSet)
x_input = x_input.reshape((1, n_steps_in, n_features)) #Suggests a single sample of n_steps?
yhat = model.predict(x_input, verbose=0)


print ("x_input is ", x_input)
print("yhat prediction is " , yhat)


# Put in graph:
#Parse x_input into 1D list - x_input_list
x_input_list = []
for i in x_input:
    for j in i:
        for k in j:
            x_input_list.append(k)

#Parse prediction sequence into 1D list - prediction_list
prediction_list = []
for i in range(len(x_input_list)-len(yhat)):
    prediction_list.append(None) #So that values appear to continue on from x_input sequence, populate list with None for length of x_input
for i in yhat:
    for j in i:
        prediction_list.append(j) #Then append values from prediction sequence yhat on it

for i in range(len(prediction_list)):
    x_input_list.append(None) #In order for GNUPlot to parse output file correctly, x_input_list needs None values appending to it to make it same length as prediction_list, 


pg.s([x_input_list, prediction_list], filename='CNN_output.out')  # save data into a file
pg.c('set title "CNN Output"; set xlabel "x-axis"; set ylabel "y-axis"')
#pg.c('set key center top')
pg.c("plot 'CNN_output.out' u 1 t 'x\_input = Test Set'")  # plot test set
pg.c("replot  'CNN_output.out' u 2  t 'yhat = Predicted value'")
