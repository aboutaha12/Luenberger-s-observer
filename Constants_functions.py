import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import EarlyStopping
from scipy import linalg
from scipy.integrate import odeint

#1-Problem data and functions:

# Constants of the problem
mu = 0.1
A = np.diag(np.array([-1, -2, -3, -4, -5, -6])) 
B = np.array([[1, 0, 1, 0, 1, 0],[0, 1, 0, 1, 0, 1]]).T

tc = 10  # since -1 is the smallest eigenvalue of A and considering that exp(-10) is nearly 0
t = np.linspace(0, tc, 1000)

# functions:
def f(x,t): # x= (x1,x2,w1,w2,w3) (x_extended) with u = w1
    return np.array([x[1], -np.sin(x[0]) - mu * x[1] + x[2], x[3], -x[4]*x[2], 0])

def f_back(x,t): # resolving the extended system backward
    sol = f(x,t)
    return -sol

def f_glob(x_glob,t): # x_glob = (x1,x2,x3,x4,x5,z1,z2,z3,z4,z5,z6).T, the global system with x and z
    y = np.array([x_glob[0],x_glob[2]])
    z = x_glob[5:]
    psi = np.array(np.dot(A,z) + np.dot(B,y))
    return np.array([x_glob[1], -np.sin(x_glob[0]) - mu * x_glob[1] + x_glob[2], x_glob[3],
         -x_glob[4]*x_glob[2], 0, psi[0], psi[1], psi[2], psi[3], psi[4], psi[5]])
