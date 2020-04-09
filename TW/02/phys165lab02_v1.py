# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:27:18 2020

@author: tomwtn
"""

import numpy as np;
from numpy.random import random as rng;
from matplotlib import pyplot as plt;
from matplotlib.animation import FuncAnimation as Fanimation;


# init val
steps = 1000;

# random +-1
def rand_array (steps):
    rand_array = np.ceil(rng(steps) * 2) * 2 - 3;
    return rand_array

# one set of x y data
def gen_data (steps):
    init1_data = rand_array(steps - 1);
    init2_data = rand_array(steps - 1);
    #print(init1_data,init2_data);
    x_data = np.zeros(steps);
    y_data = np.zeros(steps);
    for i in range(init1_data.size):
        x_data[i + 1] = x_data[i] + init1_data[i];
        y_data[i + 1] = y_data[i] + init2_data[i];
    return x_data, y_data


# begin animation
# First set up the figure, the axis, and the plot element we want to animate
fig1 = plt.figure();
ax = plt.axes(xlim = (-100, 100), ylim = (-100, 100));
line, = ax.step([], []);

# initialization function: plot the background of each frame
def init():
    line.set_data([], []);
    return line,

# animation function.  This is called sequentially
x1, y1 = gen_data(steps);  
def animate(i):
    xd = x1[0:i];
    yd = y1[0:i];
    line.set_data(xd, yd);
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = Fanimation(fig1, animate, init_func=init,
                               frames=steps, interval=1, blit=True)

plt.gca().set_aspect('equal', adjustable='box')
fig1.suptitle('Random Walk')

# save gif
#plt.rcParams["animation.convert_path"] = "D:\code\ImageMagick-7.0.10-Q16\magick.exe"
#anim.save('RandomWalk.gif', writer='imagemagick', extra_args="convert", fps=60)
plt.show()
# end animation


# begin plot
fig2, axs2 = plt.subplots(2, 2);
# data
x2, y2 = gen_data(steps);
x3, y3 = gen_data(steps);
x4, y4 = gen_data(steps);
x5, y5 = gen_data(steps);
# plots
axs2[0, 0].step(x2, y2);
axs2[0, 1].step(x3, y3);
axs2[1, 0].step(x4, y4);
axs2[1, 1].step(x5, y5);
# setting
for i in range(2):
    for j in range(2):
        axs2[i, j].set_xlim(-100,100);
        axs2[i, j].set_ylim(-100,100);
        axs2[i, j].set_aspect('equal', adjustable='box');
       
fig2.suptitle('Random Walk');
# end plot


# gen one end point
def end_point (steps):
    x_data, y_data = gen_data(steps);
    x_end = x_data[-1];
    y_end = y_data[-1];
    distance = np.sqrt(x_end ** 2 + y_end ** 2);
    return x_end, y_end, distance

# array of end points
def end_points (n, steps):
    x_arr = np.zeros(n);
    y_arr = np.zeros(n);
    r_arr = np.zeros(n);
    for i in range(n):
        x_end, y_end, distance = end_point(steps);
        x_arr[i] = x_end;
        y_arr[i] = y_end;
        r_arr[i] = distance;
    return x_arr, y_arr, r_arr

# distribution of end points
n1 = 1000;
x_end, y_end, R = end_points(n1, steps);
# plot
fig3, axs3 = plt.subplots(2, 2);
axs3[0,0].scatter(x_end, y_end);
axs3[0,1].hist(R);
axs3[1,0].hist(R**2);
axs3[1,1].hist(np.log(R));
# setting
axs3[0, 0].set_title('Distribution of end points');
axs3[0, 1].set_title('histogram of distance');
axs3[1, 0].set_title('histogram of distance^2');
axs3[1, 1].set_title('histogram of log(distance)');

#
mean_square_distance = np.mean(R**2);
print('for 1000 random walks of 1000 steps, mean_square_distance is ', mean_square_distance);
n2 = 4000;
_, _, R = end_points(n2, steps);
mean_square_distance = np.mean(R**2);
print('for 4000 random walks of 1000 steps, mean_square_distance is ', mean_square_distance);