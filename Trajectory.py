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

#functions that process:
def preprocess_x(x,min_x,max_x):
    x_prepr = np.zeros(x.shape)
    for i in range (x.shape[0]):
        x_prepr[i] = (x[i] - min_x[i])/(max_x[i] - min_x[i])
    return x_prepr

def preprocess_data_x(x_data,min_x,max_x):
    x_data_prep = np.zeros(x_data.shape)
    for i in range (x_data.shape[0]):
        x_data_prep[i,:] = preprocess_x(x_data[i,:],min_x,max_x)
    return x_data_prep

def deprocess_x(x,min_x,max_x):
    x_depro = np.zeros(x.shape)
    for i in range (x.shape[0]):
        x_depro[i] = (max_x[i] - min_x[i])*x[i] + min_x[i]
    return x_depro

def f_glob_2(x_glob,t,w): # x_glob = (x1,x2,x3,x4,x5,z1,z2,z3,z4,z5,z6).T, the global system with x and z
    y = np.array([x_glob[0],Amp*np.cos(w*t-fi)])
    z = x_glob[2:]
    psi = np.array(np.dot(A,z) + np.dot(B,y))
    return np.array([x_glob[1], -np.sin(x_glob[0]) - mu * x_glob[1] + Amp*np.cos(w*t-fi),
                     psi[0], psi[1], psi[2], psi[3], psi[4], psi[5]])

# we choose an x to stay in the domain that we fixed and on which we trained the NN
x_simu = np.array([ 0.33019075, -0.71512765]) # w1= 0.52135956 , w2=0.06146751, w3=0.10147915]
    
# the input:
T = 1/0.05076530612244898
Amp = 0.5557902677321038 
w=2*np.pi/T
fi = 0.3538347020102439 

# input : a sinusoid with a Fixed frequency:
def inpu(t):
  return Amp*np.cos(w*t-fi)

def dot_inpu(t):
  return -Amp*w*np.sin(w*t-fi)

# A new input : a sinudusoid with a variant frequency:
def omega(t,w,k1,k2):
  freqs = np.zeros(t.shape)
  print(t.shape[0])
  M = np.array([[k1**3,k1**2,k1,1],[k2**3,k2**2,k2,1],[3*k1**2,2*k1,1,0],[3*k2**2,2*k2,1,0]])
  N = np.array([w/2,w,0,0])
  x = np.linalg.solve(M,N)
  a,b,c,d = x
  for i in range(t.shape[0]):
    if t[i] < k1:
      freqs[i] = w/2
    elif t[i] > k2:
      freqs[i] = w
    else: #cad k1<= t[i] <= k2
      freqs[i] = a*(t[i]**3)+b*(t[i]**2)+c*t[i]+d
  return freqs

def dot_omega(t,w,k1,k2):
  d_freqs = np.zeros(t.shape)
  M = np.array([[k1**3,k1**2,k1,1],[k2**3,k2**2,k2,1],[3*k1**2,2*k1,1,0],[3*k2**2,2*k2,1,0]])
  N = np.array([w/2,w,0,0])
  x = np.linalg.solve(M,N)
  a,b,c,d = x
  for i in range(t.shape[0]):
    if t[i] < k1:
      d_freqs[i] = 0
    elif t[i] > k2:
      d_freqs[i] = 0
    else: #cad k1<= t[i] <= k2
      d_freqs[i] = a*(t[i]**2)+b*(t[i])+c
  return d_freqs

t = np.linspace(0,40,200)
input2 = Amp*np.cos(omega(t,w,14,26)*t-fi)
plt.plot(t,Amp*np.cos(w*t/2-fi),label='w/2')
plt.plot(t,Amp*np.cos(w*t-fi),label='w')
plt.plot(t,input2,label='input 2')
plt.legend()

def inpu_2(t,w):
  return Amp*np.cos(omega(t,w,14,26)*t-fi)

def dot_inpu_2(t,w):
  return -Amp*(dot_omega(t,w,14,26)*t+omega(t,w,14,26))*np.sin(omega(t,w,14,26)*t-fi)

def f_glob_2(x_glob,t,k1,k2,w): # x_glob = (x1,x2,x3,x4,x5,z1,z2,z3,z4,z5,z6).T, the global system with x and z
    M = np.array([[k1**3,k1**2,k1,1],[k2**3,k2**2,k2,1],[3*k1**2,2*k1,1,0],[3*k2**2,2*k2,1,0]])
    N = np.array([w/2,w,0,0])
    x = np.linalg.solve(M,N)
    a,b,c,d = x
    if t < k1:
      u = w/2
    elif t > k2:
      u = w
    else: #cad k1<= t[i] <= k2
      u = a*(t[i]**3)+b*(t[i]**2)+c*t[i]+d
    y = np.array([x_glob[0],Amp*np.cos(w*t-fi)])
    z = x_glob[2:]
    psi = np.array(np.dot(A,z) + np.dot(B,y))
    return np.array([x_glob[1], -np.sin(x_glob[0]) - mu * x_glob[1] + Amp*np.cos(w*t-fi),
                     psi[0], psi[1], psi[2], psi[3], psi[4], psi[5]])
  
  def traj_2(t,w,x=np.array([0,0])):
    x_glob_init = np.zeros((8,))
    if (x == np.array([0,0])).all():
        x1 = (2*np.pi) * np.random.random_sample() - np.pi
        x2 = 2 * np.random.random_sample() -1
        x_glob_init = np.array([x1,x2,0,0,0,0,0,0])
    
    else:
      x_glob_init[:2] = x
    print(x_glob_init[:2])

    k1 = 80
    k2 = 120

    solu = odeint(f_glob_2, x_glob_init, t, args=(k1,k2,w))
    z = solu[:,2:]
    x = solu[:,:2]

    input = inpu_2(t,w)
    dot_input = dot_inpu_2(t,w)
    freqs = omega(t,w,14,26)

    T_x = np.zeros(z.shape)
    T_x_hat = np.zeros(z.shape)

    z_modified = z.copy()
    x_predicted = np.zeros(((x.shape[0],2)))
    liste_k_1 = np.zeros(z.shape)
    for i in range (x_predicted.shape[0]):
        #u = Amp*np.cos(omega(t,w,k1,k2)[i]*t[i]-fi)
        #dot_u = -Amp*np.sin(omega(t,w,k1,k2)[i]*t[i]-fi)
        #omega = omega(t,w,k1,k2)[i]
        u= input[i]
        dot_u = dot_input[i]
        freq = freqs[i]
        x_i = x[i,:]
        x_i = np.array([x[i,0],x[i,1],u,dot_u,freq**2])
        z_i = z_modified[i,:]
        z_i = preprocess_x(z_i,min_z,max_z)
        x_predicted_i = model_predict_x.predict(z_i.reshape(1,6))[0]     
        x_predicted[i] = deprocess_x(x_predicted_i,min_x,max_x)# + np.array([2*np.pi*k1_choosed,0])

        #in order to compare z to T(x_real) and T(x_predicted:)
        p = np.array([x_predicted[i][0],x_predicted[i][1],u,dot_u,freq**2])
        x_predi_prep_i = preprocess_x(p,min_x,max_x)
        T_x_hat_i = model_predict_z.predict(x_predi_prep_i.reshape(1,5))[0] 
        T_x_hat[i] = deprocess_x(T_x_hat_i,min_z,max_z)

        x_prep = preprocess_x(x_i,min_x,max_x)
        T_x_i = model_predict_z.predict(x_prep.reshape(1,5))[0] 
        T_x[i] = deprocess_x(T_x_i,min_z,max_z)
    return x,z,x_predicted,T_x,T_x_hat,input

t = np.linspace(0,40,200)
x,z,x_predit,T_x,T_x_hat,input = traj_2(t,w,x_simu[:2])

for k in range(6):
    z_k = z[:,k]
    T_x_k = T_x[:,k]
    T_x_hat_k = T_x_hat[:,k]
    # x_k = x[:,k]
    # x_predit_k = x_predit[:,k] 

    plt.plot(t,z_k,label=f'$z_{{{k+1}}}$')
    plt.plot(t,T_x_k,label=f'$T(x_{{real}})_{{{k+1}}}$')
    #plt.plot(t,np.ones(len(t))*min_z_2[k],label=r'min_z')
    #plt.plot(t,np.ones(len(t))*mplt_z_2[k],label='mplt_z')

    plt.plot(t,T_x_hat_k,label=f'$T(x_{{predicted}})_{{{k+1}}}$')
    plt.xlabel('time')
    plt.legend(loc='upper right')
    #plt.savefig(f'essai{k}.png')
    plt.show()

for k in range (2):
    x_k = x[:,k]
    x_predit_k = x_predit[:,k] 
    plt.plot(t,x_k,label=f'$x_{{real_{{{k+1}}}}}$',color='blue')
    plt.plot(t,x_predit_k,label=f'$x_{{predicted_{{{k+1}}}}}$',color='green')
    plt.xlabel('time')
    plt.legend(loc='upper right')
    #plt.savefig(f'x{k}.png')
    plt.show()
    plt.plot(t,x_predit_k-x_k,label=f'$x_{{predicted_{{{k+1}}}}}$-$x_{{real_{{{k+1}}}}}$',color='green')
    plt.show()
