#Partly taken from https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/ 
#Sine wave prediction using CNN
import numpy as np
import PyGnuplot as pg
import math
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from numpy import array
 
################### split a univariate sequence into samples for multistep prediction########################
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
    
##################Make testSet###########################
def make_testSet(n_steps_in, n_steps_out ):
    raw_seq_SeedValue = random.random() 
    testSet = [] 
    #Create sequence for testing of seen and unseen data
    for i in range(n_steps_in + n_steps_out):
        testSet.append(math.sin(raw_seq_SeedValue))
        raw_seq_SeedValue += 0.1
    #Split it into test training set and prediction set
    unSeenData = testSet[n_steps_in:len(testSet)]
    unSeenData3 = unSeenData
    testSet = testSet[0:n_steps_in]
    #Exaggerate diff between unseen data and predicted data to show up better on graph
    unSeenData2 = []
    for i in unSeenData:
        unSeenData2.append(i*2)
    unSeenData = unSeenData2
    return testSet, unSeenData, unSeenData3
 
##################Make prediction########################
def make_prediction(testSet, n_steps_in, n_steps_out):
    x_input = array(testSet)
    x_input = x_input.reshape((1, n_steps_in, n_features)) #Suggests a single sample of n_steps?
    yhat = model.predict(x_input, verbose=0)
    #print ("x_input is ", x_input)
    #print("yhat prediction is " , yhat)
    return x_input, yhat


#################Parse results for graph##################
def parse_results(x_input, yhat, unSeenData):
    x_input_list = []     #Parse x_input into 1D list - x_input_list
    prediction_list = []  
    prediction_list2 = []  
    unSeenData_List = []
    for i in x_input:
        for j in i:
            for k in j:
                x_input_list.append(k)
                prediction_list.append(None) #So that values appear to continue on from x_input sequence, populate list with None for length of x_input
                unSeenData_List.append(None)
    #Parse prediction sequence into 1D list - prediction_list
    for i in yhat:
        for j in i:
            prediction_list.append(j) #Then append values from prediction sequence yhat on it
            prediction_list2.append(j) #Append values from prediction sequence yhat to a list that will be used for calculating error
            x_input_list.append(None)
    #Parse 
    for i in unSeenData:
        unSeenData_List.append(i)
    return x_input_list, prediction_list, prediction_list2, unSeenData_List
    
    
#################Calculate error & PA#########################
def calculate_error(prediction_list2, unSeenData3):
    rmse = np.sqrt(np.mean((array(prediction_list2) - array(unSeenData3))**2)) #Subtract the correct array of results from the predicted results array, 
    #square them so they're all positive, find the avg. of differences and then find the sqaure root of that - gives us the RMSE
    #As a %:
    PA =  100-((rmse/(np.sqrt(np.mean((array(unSeenData3))**2)))) * 100)#
    return rmse, PA


#############Put in graph:########################
def make_graph(x_input_list, prediction_list, unSeenData_List):
    pg.s([x_input_list, prediction_list, unSeenData_List], filename='CNN_output.out')  # save data into a file
    pg.c('set title "CNN Output"; set xlabel "x-axis"; set ylabel "y-axis"')
    #pg.c('set key center top')
    pg.c("plot 'CNN_output.out' u 1 w l t 'x\_input = Test Set'")  # plot test set seen data
    pg.c("replot  'CNN_output.out' u 2 w l  t 'yhat = Predicted value'") # plot test set CNN predicted data
    pg.c("replot  'CNN_output.out' u 3 w l t 'unSeenData = Correct value'") # plot test set unseen actual data


##############test_model#################################
def test_model(run_number):
    global testSet, unSeenData, unSeenData3
    global x_input_list, prediction_list, prediction_list2, unSeenData_List
    global rmse, PA
    testSet, unSeenData, unSeenData3 = make_testSet(n_steps_in, n_steps_out)
    x_input, yhat = make_prediction(testSet, n_steps_in, n_steps_out)
    x_input_list, prediction_list, prediction_list2, unSeenData_List = parse_results(x_input, yhat, unSeenData)
    rmse, PA = calculate_error(prediction_list2, unSeenData3)
    print(f"\nRun {run_number}. RMSE: {rmse} and PA: {PA}")
    return PA
    
 
#Sort input sequence:
raw_seq_SeedValue = 0.083

# define input sequence
raw_seq = []
testSet = []
x_input = []
yhat = []

# define list containers for storing output sequences
unSeenData = []
unSeenData2 = []
unSeenData3 = []
x_input_list = []
prediction_list = []  
prediction_list2 = []  
unSeenData_List = []
rmse = 0.0
PA = 0.0
PA_list = []
counter = 5

# choose a number of time steps for each sample
n_steps_in = 20
n_steps_out = 30

# Create array of values for sine wave for training
for i in range(2500):
    raw_seq.append(math.sin(raw_seq_SeedValue))
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



#Run it 5 times to compare errors:
for i in range(counter):
    PA_list.append(test_model(i))
print("Average PA from " + counter + " runs is " + np.mean(array(PA_list)))