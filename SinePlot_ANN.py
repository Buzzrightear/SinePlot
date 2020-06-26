#from https://www.guru99.com/pytorch-tutorial.html
#Attempt at linear regression, which doesn't work because linear regression can't work on a sine wave I think
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.layer = torch.nn.Linear(1, 1)

   def forward(self, x):
       x = self.layer(x)
       return x

net = Net()
#print(net)

seedValue = 0.083
trainingSet = []
x =[]

#Create list of values for sine wave
for i in range(250):
    trainingSet.append(math.sin(seedValue))
    x.append(seedValue)
    seedValue += 0.1
y = trainingSet

x = np.asarray(x)
y = np.asarray(trainingSet)

x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()

# Define Optimizer and Loss Function
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

inputs = Variable(x)
outputs = Variable(y)
for i in range(100):
   prediction = net(inputs)
   loss = loss_func(prediction, outputs)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   if i % 10 == 0:
       # plot and show learning process
       plt.cla()
       plt.scatter(x.data.numpy(), y.data.numpy())
       plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
       #plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
       plt.pause(0.5)

plt.show()

