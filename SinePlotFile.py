# Author: Henry Stanton
# Date created: 16/06/2020

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

trainingSetSeedValue = 0.1
testSetSeedValue = 0.0

trainingSet = []
testSet = []

# Create array of values for sine wave
for i in range(100):
    trainingSet.append(math.sin(trainingSetSeedValue))
    trainingSetSeedValue += 0.1

for i in range(100):
    testSet.append(math.sin(trainingSetSeedValue))
    trainingSetSeedValue += 0.1


# plot values
    """#plt.plot(trainingSet, color = 'red', marker = "o")
    plt.plot(testSet, color = 'green', marker = "x")
    plt.title("Some values of sine")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show() """
print(trainingSet)
trainingSet = torch.FloatTensor(trainingSet).view(-1)
print(trainingSet)
train_window = 10
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(trainingSet, train_window)
print(len(train_inout_seq))

print(train_inout_seq[:5])

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]