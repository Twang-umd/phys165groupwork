# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:31:45 2020

@author: tomwtn
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#working dir
path = ('D:\code\python\working');
os.chdir(path);
#read data
for fname in os.listdir():
    if fname.endswith('1.csv'):
        X = np.loadtxt(fname,delimiter=',');
    elif fname.endswith('2.csv'):
        Y = np.loadtxt(fname,delimiter=',');
X=X.T
Y=Y.T

# function V(t)
def V(t,A,tau):  
    output = A*( 1 - np.exp(-1*t/tau) );
    return output

# function W(t)
def W(t,A,tau):  
    output = A*( np.exp(-1*t/tau) - 1 + (t/tau) );
    return output

#color
C = ['b','g','r','c','m','y','k','b','g'];
#linetp
linetp = ['-','*','^']*3;

#
import matplotlib;
matplotlib.rc('text', usetex=True);#latex like

#domain
t = np.linspace(0,2,101);

#V(t)
fig1 = plt.figure(1);
for A in np.linspace(1,3,3):
    for tau in np.linspace(1,3,3):
        linetype = C[(int(A)-1)*3+int(tau)-1]+linetp[(int(A)-1)*3+int(tau)-1];
        lb = 'A='+str(A)+' '+r'$\tau$='+str(tau);
        plt.plot(t,V(t,A,tau),linetype,label=lb);
#gca
grph1 = plt.gca();
#setting
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., fontsize=24);
plt.title(r'$V(t)=A\left[1-e^{-t/\tau}\right]$', fontsize=24);
plt.xlabel(r'time', fontsize=24);
plt.ylabel(r'bacterial population', fontsize=24);
plt.tick_params(labelsize=24);
plt.axis([0, 2, 0, 2])

#fit
fig3 = plt.figure(3)
d1 = np.linspace(0,7,701);
#W(t)
plt.plot(d1,V(d1,1,3),'r-',label='theory $A=1$ '+r'$\tau=3$');
#exp data
plt.plot(X[0][:],X[1][:],'go',label='exp data');
#setting
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., fontsize=24);
plt.title(r'$V(t)=A\left[1-e^{-t/\tau}\right]$', fontsize=24);
plt.xlabel(r'time', fontsize=24);
plt.ylabel(r'bacterial population', fontsize=24);
plt.tick_params(labelsize=24);
plt.axis([0, 7, 0, 1]);

#W(t)
fig2 = plt.figure(2);
for A in np.linspace(1,3,3):
    for tau in np.linspace(1,3,3):
        linetype = C[(int(A)-1)*3+int(tau)-1]+linetp[(int(A)-1)*3+int(tau)-1];
        lb = 'A='+str(A)+' '+r'$\tau=$'+str(tau);
        plt.plot(t,W(t,A,tau),linetype,label=lb);
#setting
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., fontsize=24);
plt.title(r'$V(t)=A\left[e^{-t/\tau}-1+t/\tau\right]$', fontsize=24);
plt.xlabel(r'time', fontsize=24);
plt.ylabel(r'bacterial population', fontsize=24);
plt.tick_params(labelsize=24);
plt.axis([0, 2, 0, 2])

#fit
fig4 = plt.figure(4)
d2 = np.linspace(0,30,1501);
#W(t)
plt.plot(d2,W(d2,1.5,30),'r-',label='theory $A=2$ '+r'$\tau=40$');
#exp data
plt.plot(Y[0][:],Y[1][:],'go',label='exp data');
#setting
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., fontsize=24);
plt.title(r'$V(t)=A\left[e^{-t/\tau}-1+t/\tau\right]$', fontsize=24);
plt.xlabel(r'time', fontsize=24);
plt.ylabel(r'bacterial population', fontsize=24);
plt.tick_params(labelsize=24);
plt.axis([0, 30, 0, 0.5]);

#show plots
plt.show()

