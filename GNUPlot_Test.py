import numpy as np
import PyGnuplot as pg
import math
from numpy import array
x = np.arange(1000)/20.0 #creates array of items between 0 and 50 with a step of 0.05. Same can be done with x = np.arrange(0,50,0.05)

y1 = x-25 #returns x array but with each value having had 25 subtracted from it
y2 = y1*np.sin(x-25) #returns array y1 but each value has had sin(s-25) applied to it

#testing what comes out
raw_seq_SeedValue = 0.083
testSet = []
n_steps = 100
for i in range(n_steps):
    testSet.append(math.sin(raw_seq_SeedValue))
    raw_seq_SeedValue += 0.1
x_input = array(testSet)
x_input = x_input.reshape((1, n_steps, 1))
#print(x_input)

x_input_list = []
for i in x_input:
    for j in i:
        for k in j:
            x_input_list.append(k)
print(x_input_list)

pg.s([x_input_list], filename='example1.out')  # save data into a file
pg.c('set title "example1"; set xlabel "x-axis"; set ylabel "y-axis"')
pg.c('set key left top')
pg.c("plot 'example1.out'  t 'Test key'")  # plot first part




'''
pg.s([x, y1, y2], filename='example.out')  # save data into a file t.out
pg.c('set title "example.pdf"; set xlabel "x-axis"; set ylabel "y-axis"')
#pg.c('set yrange [-50.0:25.0]; set key center top')
pg.c('set key left top')

pg.c("plot 'example.out'  w l t 'Test title'")  # plot first part

#pg.c("plot 'example.out' u 1:2 w l t 'y=x-25")  # The using directive tells the plot command which columns to use. plot "data" using 1:2 selects the first column for the x axis and the second column for the y axis.

pg.c("replot 'example.out'  u 1:3 w l t 'y=(x-25)*sin(x-25)'")
#pg.c("replot 'example.out' u 1:(-$2) w l t 'y=25-x'")

#pg.pdf('example.pdf')  # export figure into a pdf file

'''
