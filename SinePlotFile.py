# Author: Henry Stanton
# Date created: 16/06/2020
#Taken from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

trainingSetSeedValue = 0.083
testSetSeedValue = 0.0

trainingSet = []
testSet = []

# Create array of values for sine wave
for i in range(1000):
    trainingSet.append(math.sin(trainingSetSeedValue))
    trainingSetSeedValue += 0.1

for i in range(1000):
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
train_window = 100
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

#Create training model
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#Make predictions
fut_pred = 200

test_inputs = trainingSet[-train_window:].tolist()
print(test_inputs)

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())



x = np.arange(132, 144, 1)
print(x)

#print(len(trainingSet[-train_window:]))
print(len(testSet))
print(len(test_inputs[train_window:]))

plt.title('Values of sine')
plt.grid(True)
#plt.autoscale(axis='x', tight=True)
#plt.plot(trainingSet[train_window:], label='training set')
plt.plot(testSet[train_window:],label='test set')
plt.plot(test_inputs[train_window:], label='prediction set')
plt.legend()
plt.show()