# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:17:17 2020

@author: tomwtn
"""

import numpy as np;
from numpy import pi as pi;
import matplotlib.pyplot as plt;
from scipy.integrate import solve_ivp as ODEINT;

# parameter
def parameter (beta, omega_0, omega_D, A):
    b = beta;
    o0 = omega_0;
    oD = omega_D;
    a = A;
    return b, o0, oD, a

# domain
def domain ():
    ini_t = 0;
    end_t = 10;
    dt = 0.01;
    NoP = int( (end_t - ini_t) / dt + 1 );
    t = np.linspace(ini_t, end_t, NoP);
    return ini_t, end_t, dt, NoP, t

# DY/dt
def DY (t, y, A, omega_0, omega_D, beta):
    ypp = A * np.cos(omega_D * t) - beta * y[1] - (omega_0 ** 2) * y[0];
    Dy = [y[1], ypp];
    return Dy

# init Y
def init_Y (y0, yp0):
    initY = [y0, yp0];
    return initY

# undriven + undamped

def UnDri_UnDam ():
    # parameter
    beta, omega_0, omega_D, A = parameter(np.zeros(5),
                                          [0.5*pi,1*pi,1.5*pi,2*pi,2.5*pi],
                                          np.zeros(5),
                                          np.zeros(5));
    # domain
    _, _, _, _, t = domain();
    # init Y
    initY = init_Y(1, 0);
    # plot
    fig1 = plt.figure(1);
    for i in range(A.size):
        sol = ODEINT(lambda t, y: DY(t, y, A[i], omega_0[i], omega_D[i], beta[i]),
                     [t[0], t[-1]], initY, t_eval=t);
        lb = r'$\omega_{0}=$' + str(0.5*(i + 1)) + r'$\pi$';
        plt.plot(sol.t, sol.y[0,:], label = lb);
    # plot setting
    plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$d^{2}y/dt^{2} = - \omega_{0}^{2}y$', fontsize=24);
    plt.xlabel(r't', fontsize=24);
    plt.ylabel(r'y', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 5, -1, 1])
    return ()
UnDri_UnDam();

# damped

def Damped ():
    # parameter
    beta, omega_0, omega_D, A = parameter([0,1*pi,10*pi],
                                          [1*pi,1*pi,1*pi],
                                          np.zeros(3),
                                          np.zeros(3));
    # domain
    _, _, _, _, t = domain();
    # init Y
    initY = init_Y(1, 0);
    # plot
    fig2 = plt.figure(2);
    for i in range(A.size):
        sol = ODEINT(lambda t, y: DY(t, y, A[i], omega_0[i], omega_D[i], beta[i]),
                     [t[0], t[-1]], initY, t_eval=t);
        lb = r'$\beta^{2} - \omega_{0}^{2} = $' + str(beta[i]**2-omega_0[i]**2);
        plt.plot(sol.t, sol.y[0,:], label = lb);
    # plot setting
    plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$d^{2}y/dt^{2} = - \omega_{0}^{2}y - \beta dy/dt$', fontsize=24);
    plt.xlabel(r't', fontsize=24);
    plt.ylabel(r'y', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 10, -1, 1])
    return ()
Damped();

# driven

def Driven ():
    # parameter
    beta, omega_0, omega_D, A = parameter(np.zeros(5),
                                          [1*pi,1*pi,1*pi,1*pi,1*pi],
                                          [0,0.5*pi,1*pi,1.5*pi,2*pi],
                                          [5,5,5,5,5]);
    # domain
    _, _, _, _, t = domain();
    # init Y
    initY = init_Y(1, 0);
    # plot
    fig3 = plt.figure(3);
    for i in range(beta.size):
        sol = ODEINT(lambda t, y: DY(t, y, A[i], omega_0[i], omega_D[i], beta[i]),
                     [t[0], t[-1]], initY, t_eval=t);
        lb = r'$\omega_{D} = $' + str(i*0.5) + r'$\omega_{0}$';
        plt.plot(sol.t, sol.y[0,:], label = lb);
    # plot setting
    plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$d^{2}y/dt^{2} = - \omega_{0}^{2}y + A\cos(\omega_{D}t)$', fontsize=24);
    plt.xlabel(r't', fontsize=24);
    plt.ylabel(r'y', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 6, -10, 10]);
    
    # A-f
    fig4 = plt.figure(4);
    def Amp (omega_0, omega_D, beta):
        D = omega_0**2 / (np.sqrt( (omega_0**2 - omega_D**2)**2 +  omega_D**2 * beta**2 ));
        return (D)
    omega_0 = np.linspace(0, 10*pi, 1001);
    OD = np.linspace(0, 9, 10)*pi;
    for i in OD:
        Y = Amp(omega_0, i, 3);
        lb = r'$\omega_{D} =$' + str(i/pi) + r'$\pi$';
        plt.plot(omega_0, Y, label = lb);
    plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$Amplitude-\omega_{0}\ Plot,A=1,\beta=1$', fontsize=24);
    plt.xlabel(r'$\omega_{0}$', fontsize=24);
    plt.ylabel(r'Amplitude', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 10*pi, 0, 10]);
    
    # theta-f
    fig5 = plt.figure(5);
    def Phase (omega_0, omega_D, beta):
        D = 1 - np.arccos( (omega_0**2 - omega_D**2) / np.sqrt((omega_0**2 - omega_D**2)**2 +  omega_D**2 * beta**2) ) / pi;
        return (D)
    omega_0 = np.linspace(0, 10*pi, 1001);
    OD = np.linspace(0, 9, 10)*pi;
    for i in OD:
        Y = Phase(omega_0, i, 1);
        lb = r'$\omega_{D} =$' + str(i/pi) + r'$\pi$';
        plt.plot(omega_0, Y, label = lb);
    plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$Phase-\omega_{0}\ Plot,A=1,\beta=1$', fontsize=24);
    plt.xlabel(r'$\omega_{0}$', fontsize=24);
    plt.ylabel(r'$Phase/\pi$', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 10*pi, 0, 1]);
    return ()
Driven();

# driven + damped

def DrivenDamped ():
    # parameter
    beta, omega_0, omega_D, A = parameter([1,1,1,5,5,5],
                                          [1*pi,1*pi,1*pi,1*pi,1*pi,1*pi],
                                          [0.1*pi,1*pi,2*pi,0.1*pi,1*pi,2*pi],
                                          [5,5,5,5,5,5]);
    # domain
    _, _, _, _, t = domain();
    # init Y
    initY = init_Y(1, 0);
    # plot
    fig6 = plt.figure(6);
    for i in range(6):
        sol = ODEINT(lambda t, y: DY(t, y, A[i], omega_0[i], omega_D[i], beta[i]),
                     [t[0], t[-1]], initY, t_eval=t);
        lb = r'$\omega_{D} = $' + str(omega_D[i]/omega_0[i]) + r'$\omega_{0},\beta=$' + str(beta[i]);
        plt.plot(sol.t, sol.y[0,:], label = lb);
    # plot setting
    plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left', borderaxespad=0., fontsize=24);
    plt.title(r'$d^{2}y/dt^{2} = - \omega_{0}^{2}y - \beta dy/dt + A\cos(\omega_{D}t)$', fontsize=24);
    plt.xlabel(r't', fontsize=24);
    plt.ylabel(r'y', fontsize=24);
    plt.tick_params(labelsize=24);
    plt.axis([0, 6, -5, 5]);
DrivenDamped();
