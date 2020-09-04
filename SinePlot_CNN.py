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
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])



