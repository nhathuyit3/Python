from __future__ import division, print_function, unicode_literals
import math
import numpy as np 

import matplotlib.pyplot as plt
x0 = -1

def cost(x):
    return -x**3 + 3*x**2 - 4*x + 1

def grad(x):
    return -3*x**2 + 6*x - 4   

def myGD1(x0, eta = 0.1):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    x = np.asarray(x)
    return (x, it)

def plot_fn(fn, xmin = -5, xmax = 5, xaxis = True, opts = 'b-'):
    x = np.linspace(xmin, xmax, 1000)
    y = fn(x)
    ymin = np.min(y) - .5
    ymax = np.max(y) + .5
    plt.axis([xmin, xmax, ymin, ymax])
    if xaxis:
        x0 = np.linspace(xmin, xmax, 2)
        plt.plot([xmin, xmax], [0, 0], 'k')
    plt.plot(x, y, opts)
# plot_fn(cost, -10, 10)
## Momentum example
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
def viz_alg_1d(x, cost, filename = 'momentum1d2.gif'):
#     x = x.asarray()
    it = len(x)
    y = cost(x)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    
    xmin, xmax = -4, 6
    ymin, ymax = -12, 25
    
    x0 = np.linspace(xmin-1, xmax+1, 1000)
    y0 = cost(x0)
       
    fig, ax = plt.subplots(figsize=(4, 4))  
    
    def update(i):
        ani = plt.cla()
        plt.axis([xmin, xmax, ymin, ymax])
        plt.plot(x0, y0)
        ani = plt.title('$f(x) = -x^3 + 3x^2 - 4x + 1; x_0 = -1; \eta = 0.1; \gamma = 0.9$')
        if i == 0:
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        else:
            ani = plt.plot(x[i-1], y[i-1], 'ok', markersize = 7)
            ani = plt.plot(x[i-1:i+1], y[i-1:i+1], 'k-')
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        label = 'GD with Momemtum: iter %d/%d' %(i, it)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    anim.save(filename, dpi = 100, writer = 'imagemagick')
    plt.show()
    
# x = np.asarray(x)
(x, it) = myGD1(-5, 0.01)
viz_alg_1d(x, cost)

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
def viz_alg_1d_2(x, cost, filename = 'nomomentum1d.gif'):
#     x = x.asarray()
    it = len(x)
    y = cost(x)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = -4, 6
    ymin, ymax = -12, 25
    x0 = np.linspace(xmin-1, xmax+1, 1000)
    y0 = cost(x0)
       
    fig, ax = plt.subplots(figsize=(4, 4))  
    
    def update(i):
        ani = plt.cla()
        plt.axis([-4 , 6, -13, 26])
        plt.plot(x0, y0)
        plt.axis([xmin, xmax, ymin, ymax])
        ani = plt.title('$f(x) = -x^3 + 3x^2 - 4x + 1; x_0 = -1; \eta = 0.1$')
        if i == 0:
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        else:
            ani = plt.plot(x[i-1], y[i-1], 'ok', markersize = 7)
            ani = plt.plot(x[i-1:i+1], y[i-1:i+1], 'k-')
            ani = plt.plot(x[i], y[i], 'ro', markersize = 7)
        label = 'GD without Momemtum: iter %d/%d' %(i, it)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    anim.save(filename, dpi = 100, writer = 'imagemagick')
    plt.show()
    
# x = np.asarray(x)
(x, it) = myGD1(5, 0.1)
viz_alg_1d_2(x, cost)