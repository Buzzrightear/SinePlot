import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg') #https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable 
import matplotlib.pyplot as plt
import math


seedValue = 0.083


trainingSet = [[]]
testSet = [[]]

# Create array of values for sine wave
for i in range(200):
    trainingSet.append([seedValue,math.sin(seedValue)])
    seedValue += 0.1

for i in range(200):
    testSet.append([seedValue,math.sin(seedValue)])
    seedValue += 0.1

df_training = pd.DataFrame(trainingSet)
df_test = pd.DataFrame(testSet)

df_training.columns=['val1','val2']




rolling_mean1 = df_training.val2.rolling(window=20).mean()
rolling_mean2 = df_training.val2.rolling(window=50).mean()
plt.plot(df_training.val1, rolling_mean1, label='rolling mean - window size: 20', color='orange')
plt.plot(df_training.val1, rolling_mean2, label='rolling mean - window size: 50', color='red')
plt.plot(df_training.val1, df_training.val2, label='training set', color='blue')

plt.legend(loc='upper left')
plt.show()
