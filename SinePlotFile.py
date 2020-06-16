# Author: Henry Stanton
# Date created: 16/06/2020
import math
import matplotlib.pyplot as plt

x = 0.1
y = 0.9

valuesListX = []
valuesListY = []

# Create array of values for sine wave
for i in range(200):
    valuesListX.append(math.sin(x))
    valuesListY.append(math.sin(y))
    x += 0.1
    y += 0.1

# plot values
plt.plot(valuesListX, color = 'red', marker = "o")
plt.plot(valuesListY, color = 'green', marker = "x")
plt.title("Some values of sine")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

