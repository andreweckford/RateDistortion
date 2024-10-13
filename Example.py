#!/usr/bin/env python
# coding: utf-8

# Example illustrating the use of the RateDistortion module

import numpy as np
import matplotlib.pyplot as plt
from RateDistortion import *

# Parameters

# prior probability of x, which is ternary
px = np.array([0.3,0.4,0.3])

# distortion function - in this example, there are two possible elements of y for each x
dxy = np.array([[2,0,1],[0,2,1]])

# In the final cell, we display the solution p(x|y) for D nearest this target
D_target = 0.5


rd = getRD(px,dxy)

R = rd['r_v']
D = rd['Dmax_v']
plt.plot(D,R)

ax = plt.gca()
ax.set_ylabel('$R(D)$ (nats / channel use)')
ax.set_xlabel('Average distortion $D$')

i = np.sum(D < D_target)

# joint probability of x and y in matrix form
ny = np.shape(dxy)[0] # size of y alphabet
nx = np.shape(dxy)[1] # size of x alphabet
pxy = unstack(rd['p'][i],ny,nx) 

p_y_given_x = pxy @ np.linalg.inv(np.diag(px))
print(p_y_given_x)

# as this is a probability of y given x, the vertical sums should be equal to 1
# uncomment the following line to see this
# print(np.sum(p_y_given_x,axis=0))

print('Close the plot window to terminate the program.')

plt.show()
