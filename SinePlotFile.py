# Author: Henry Stanton
# Date created: 16/06/2020
import math
import matplotlib .pyplot as plt

x = 0.1
valuesList = []

# Create array of values for sine wave
for i in range(200):
    valuesList.append(math.sin(x))
    x += 0.1

# plot values


