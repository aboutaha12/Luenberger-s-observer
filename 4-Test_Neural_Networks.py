import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import EarlyStopping
from scipy import linalg
from scipy.integrate import odeint

#Problem data and functions:

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

#Test of the models

min_x=np.load("min_x_train")
max_x=np.load("max_x_train")
min_z=np.load("min_z_train")
max_z=np.load("max_z_train")

x_test_preprocessed = np.load("x_test_preproc.npy")
z_test_preprocessed = np.load("z_test_preproc.npy")

model_predict_z =  tf.keras.models.load_model("mymodel_-2,2")
model_predict_x = tf.keras.models.load_model("mymodel_inv_-2,2")

#Test of the model_predict_x on the test set
results_prep = model_predict_x.evaluate(z_test_preprocessed,x_test_preprocessed[:,:2],batch_size=128)
print(results_prep)

# Test on random points in the test set
i = np.random.randint(0,640000,4)
for k in i:
  print(f'modele prep k : {k}, x_test : {deprocess_x(x_test_preprocessed[k,:2],min_x,max_x)}, x_predicted :{deprocess_x(model_predict_x.predict(z_test_preprocessed[k,:].reshape(1,6))[0],min_x,max_x)}')
  print('\n')

#Test of model_predict_x on random points


def test_x(tc):
    x1 = (2*np.pi) * np.random.random_sample((5,)) - np.pi
    x2 = 2 * np.random.random_sample((5,)) -1
    x3 = 2 * np.random.random_sample((5,)) -1
    x4 = 2 * np.random.random_sample((5,)) -1
    x5 = (1-0.1) * np.random.random_sample((5,)) + 0.1
    s = 0
    x_errors = np.zeros((5**5,2))
    for i_x1 in x1:
        for i_x2 in x2:
            for i_x3 in x3:
                for i_x4 in x4:
                    for i_x5 in x5:
                        x_init = np.array([i_x1, i_x2, i_x3, i_x4, i_x5])

                        # Simulate (1) in backward during tc:
                        sol = odeint(f_back, x_init, tc)
                        x_init_back = sol[-1,:] # on prend le dernier point 

                        # Simulate (1) et (2) forward during tc:
                        x_glob_init = np.array([x_init_back[0], x_init_back[1],x_init_back[2],x_init_back[3],x_init_back[4],
                                0,0,0,0,0,0])
                        solu = odeint(f_glob, x_glob_init, tc)
                        z_reel = solu[-1,5:]
                        z_reel_preproc = preprocess_x(z_reel,min_z,max_z)

                        # x predicted by the neural network:
                        x_predicted = deprocess_x(model_predict_x.predict(z_reel_preproc.reshape(1,6))[0],min_x,max_x) 

                        # compute the error in the array:
                        x_errors[s,:] = x_init[:2] - x_predicted
                        s += 1
                                   
    return x_errors

tc = 10
t_c = np.linspace(0, tc, 1000)
x_errors = test_x(t_c)

# Plot an histogram of the errors:
def hist(z_errors,ligne,colonne):
    fig = plt.figure(figsize=(10,6))
    for i in range(z_errors.shape[1]):
        ax = plt.subplot(ligne,colonne,i+1)

        z_i = np.abs(z_errors[:,i])
        for k in range(len(z_i)):
            if z_i[k] == 0:
                z_i[k] = 10**(-8)
        z_i = np.log10(z_i) # logarithmic scale

        bins = np.arange(0,1,0.1)
        ax.hist(z_i)
        ax.set_title(f"Errors for the {i}-th composent")
    plt.xlabel(f'mean squared errors')
    plt.ylabel('number of points')
    plt.show()
    
#hist(x_errors,1,2) # uncomment to show the histogramms of the errors on x

#Test of model_predict_z on the test set
results_prep = model_predict_z.evaluate(x_test_preprocessed,z_test_preprocessed,batch_size=128)
print(results_prep)

# Test on random points in the test set
i = np.random.randint(0,640000,4)
for k in i:
  print(f'modele prep k : {k}, z_test : {deprocess_x(z_test_preprocessed[k,:],min_z,max_z)}, z_predict:{deprocess_x(model_predict_z.predict(x_test_preprocessed[k,:].reshape(1,5))[0],min_z,max_z)}')
  print('\n')

#Test of model_predict_z on random points

def test_z(tc):
    x1 = (2*np.pi) * np.random.random_sample((5,)) - np.pi
    x2 = 2 * np.random.random_sample((5,)) -1
    x3 = 2 * np.random.random_sample((5,)) -1
    x4 = 2 * np.random.random_sample((5,)) -1
    x5 = (1-0.1) * np.random.random_sample((5,)) + 0.1
    s = 0
    z_errors = np.zeros((5**5,6))
    for i_x1 in x1:
        for i_x2 in x2:
            for i_x3 in x3:
                for i_x4 in x4:
                    for i_x5 in x5:
                        x_init = np.array([i_x1, i_x2, i_x3, i_x4, i_x5])

                        # Simulate (1) backward during tc:
                        sol = odeint(f_back, x_init, tc)
                        x_init_back = sol[-1,:] # on prend le dernier point 

                        # Simulate (1) et (2) forward during the same tc:
                        x_glob_init = np.array([x_init_back[0], x_init_back[1],x_init_back[2],x_init_back[3],x_init_back[4],
                                0,0,0,0,0,0])
                        solu = odeint(f_glob, x_glob_init, tc)
                        z_real = solu[-1,5:]
                        x_preproc = preprocess_x(x_init,min_x,max_x)

                        # z_predicted using the neural network:
                        z_predicted = deprocess_x(model_predict_z.predict(x_preproc.reshape(1,5))[0],min_z,max_z) 

                        # compute the error:
                        z_errors[s,:] = z_real - z_predicted 
                        s += 1
                                   
    return z_errors

# Array of the errors
tc = 10
t_c = np.linspace(0, tc, 1000)
z_errors = test_z(t_c)

# Histogram of the errors
# hist(z_errors,3,2) # uncomment to show the histogramms of the errors on z

