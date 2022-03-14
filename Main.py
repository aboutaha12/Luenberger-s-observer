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

#2-Creation of the datasets

'''inits = np.zeros((20**5,11)) # array of the 20**5 points obtained with the integration backward of (1) during tc
finals = np.zeros((20**5,11)) # array of the 20**5 points with the right couples (x,T(x))

# uniform distributions
x1_boucle = np.linspace(-np.pi, np.pi, 20)
x2_boucle = np.linspace(-2, 2, 20)
x3_boucle = np.linspace(-1, 1, 20)
x4_boucle = np.linspace(-1, 1, 20)
x5_boucle = np.linspace(0.1, 1, 20)'''

'''s = 0 # counter 

for i_x1 in x1_boucle:
    for i_x2 in x2_boucle:
        for i_x3 in x3_boucle:
            for i_x4 in x4_boucle:
                for i_x5 in x5_boucle:
                    x_init = np.array([i_x1, i_x2, i_x3, i_x4, i_x5])

                    # Simuler (1) en temps rétrograde pendant tc:
                    sol = odeint(f_back, x_init, t)
                    x_init_back = sol[-1,:] # on prend le dernier point 

                    # # Simuler (1) et (2) en temps forward pendant tc
                    # x_glob_init = np.array([x_init_back[0], x_init_back[1],x_init_back[2],x_init_back[3],x_init_back[4],
                    #         0,0,0,0,0,0])
                    # solu = odeint(f_glob, x_glob_init, t)
                    # point = solu[-1,:]

                    # rajouter le point dans l'array:
                    inits[s,:5] = x_init_back
                    s += 1'''

# np.save("dataset_init.npy", inits)

'''inits = np.load("dataset_init.npy")'''

'''for i in range(20**5):
    #Integer (1) et (2) forward during tc
    solu = odeint(f_glob, inits[i,:], t)
    point = solu[-1,:]

    # Add the point to the array:
    finals[i,:] = point'''

# save the array:
#np.save("dataset_final.npy", finals)

# The final dataset with the couples (x,T(x))
#finals = np.load("dataset_final.npy")

#3-Train and test sets

'''loaded_array = np.load("dataset_final.npy")
x = loaded_array[:,:5]
z = loaded_array[:,5:]'''

# split into train and test sets
'''x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.20, random_state=42)
np.save("x_train_3.npy", x_train)
np.save("z_train_3.npy", z_train)
np.save("x_test_3.npy", x_test)
np.save("z_test_3.npy", z_test)'''

#4-Preprocess the data

x_train = np.load("x_train_3.npy")
z_train =np.load("z_train_3.npy")
x_test =np.load("x_test_3.npy")
z_test =np.load("z_test_3.npy")

# Using the MinMax Scaling:
min_x = np.zeros((5,))
max_x = np.zeros((5,))
min_z = np.zeros((6,))
max_z = np.zeros((6,))
for i in range(5):
    min_x[i] = np.min(x_train[:,i])
    max_x[i] = np.max(x_train[:,i])
    min_z[i] = np.min(z_train[:,i])
    max_z[i] = np.max(z_train[:,i])
min_z[5] = np.min(z_train[:,5])
max_z[5] = np.max(z_train[:,5])

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

x_train_preprocessed = preprocess_data_x(x_train,min_x,max_x)
x_test_preprocessed = preprocess_data_x(x_test,min_x,max_x)
z_train_preprocessed = preprocess_data_x(z_train,min_z,max_z)
z_test_preprocessed = preprocess_data_x(z_test,min_z,max_z)

'''
np.save("x_train_preproc_2.npy", x_train_preprocessed)
np.save("x_test_preproc_2.npy", x_test_preprocessed)
np.save("z_train_preproc_2.npy", z_train_preprocessed)
np.save("z_test_preproc_2.npy", z_test_preprocessed) 
'''

'''x_train_preprocessed = np.load("x_train_preproc_2.npy")
x_test_preprocessed = np.load("x_test_preproc_2.npy")
z_train_preprocessed = np.load("z_train_preproc_2.npy")
z_test_preprocessed = np.load("z_test_preproc_2.npy")'''
x_train_2 = x_train_preprocessed[:,:2] # we only want to predict (x1,x2)

#5-Training of the neural networks

def train1(): # 1st neural network to predict z=T(x)
    #Creation of the model
    ES = EarlyStopping(monitor='loss',patience=5)
    model = Sequential() 
    input_layer = tf.keras.Input(shape=(5,))
    model.add(input_layer) 
    hidden_layer1 = Dense(64,activation='relu') 
    model.add(hidden_layer1)
    hidden_layer2 = Dense(64,activation='relu')
    model.add(hidden_layer2)
    hidden_layer3 = Dense(32,activation='relu')
    model.add(hidden_layer3)
    output_layer = Dense(6)
    model.add(output_layer)
    #compiling the sequential model
    model.compile(optimizer = 'adam',
                  loss = 'mse',
                  metrics=['mse','mae'])
    model.summary()
    history = model.fit(x_train_preprocessed, z_train_preprocessed, epochs=30, callbacks=[ES])
    # plot metrics
    plt.plot(history.history['mse'])
    plt.show()
    model.save('mymodel_-2,2')

def train2(): #2nd neural network to predict x=T*(z) where T* is the inverse of T
    #Creation of the model
    ES = EarlyStopping(monitor='loss',patience=5)
    model = Sequential() 
    input_layer = tf.keras.Input(shape=(6,))
    model.add(input_layer) 
    hidden_layer1 = Dense(64,activation='relu') 
    model.add(hidden_layer1)
    hidden_layer2 = Dense(64,activation='relu')
    model.add(hidden_layer2)
    hidden_layer3 = Dense(32,activation='relu')
    model.add(hidden_layer3)
    output_layer = Dense(2)
    model.add(output_layer)
    #compiling the sequential model
    model.compile(optimizer = 'adam',
                  loss = 'mse',
                  metrics=['mse','mae'])
    model.summary()
    history = model.fit(z_train_preprocessed,x_train_2 , epochs=50, callbacks=[ES])
    # plot metrics
    plt.plot(history.history['mse'])
    plt.show()
    model.save('mymodel_inv_-2,2')

train2()

#6-Test of the models

model_predict_z =  tf.keras.models.load_model("mymodel_-2,2")
model_predict_x = tf.keras.models.load_model("mymodel_inv_-2,2")

**Test of the model_predict_x on the test set**


results_prep = model_predict_x.evaluate(z_test_preprocessed,x_test_preprocessed[:,:2],batch_size=128)
print(results_prep)

i = np.random.randint(0,640000,4)
for k in i:
  print(f'modele prep k : {k}, x_test : {deprocess_x(x_test_preprocessed[k,:2],min_x,max_x)}, x_predicted :{deprocess_x(model_predict_x.predict(z_test_preprocessed[k,:].reshape(1,6))[0],min_x,max_x)}')
  print('\n')

**Test of model_predict_x on random points**


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
                        x_predit = deprocess_x(model_predict_x.predict(z_reel_preproc.reshape(1,6))[0],min_x,max_x) 

                        # compute the error in the array:
                        x_errors[s,:] = x_init[:2] - x_predit 
                        s += 1
                                   
    return x_errors

tc = 10
t_c = np.linspace(0, tc, 1000)
x_errors = test_x(t_c)

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
    

def hist_i(z_errors,ligne,colonne,i,bol=False):
    fig = plt.figure(figsize=(10,6))
    z_i = np.abs(z_errors[:,i])
    for k in range(len(z_i)):
        if z_i[k] == 0:
            z_i[k] = 10**(-8)
    z_i = np.log10(z_i) # logarithmic scale

    bins = np.arange(0,1,0.1)
    plt.hist(z_i)
    if not bol:
      #plt.title(f"Errors for ($z_{{{i}}}$,$\hat{{z}}_{{{i}}}$)")
      plt.xlabel(f'mean squared errors')
      plt.ylabel('number of points')
      #plt.savefig(f'histogramme_z_{i}')
    else:
      #plt.title(f"Errors for ($x_{{{i}}}$,$\hat{{x}}_{{{i}}}$)")
      plt.xlabel(f'mean squared errors')
      plt.ylabel('number of points')
      #plt.savefig(f'histogramme_x_{i}') 
    plt.show()

hist(x_errors,1,2)

for i in range(x_errors.shape[1]): 
    hist_i(x_errors,1,2,i,True)

**Test of model_predict_z on the test set**

results_prep = model_predict_z.evaluate(x_test_preprocessed,z_test_preprocessed,batch_size=128)
print(results_prep)

i = np.random.randint(0,640000,4)
for k in i:
  print(f'modele prep k : {k}, z_test : {deprocess_x(z_test_preprocessed[k,:],min_z,max_z)}, z_predict:{deprocess_x(model_predict_z.predict(x_test_preprocessed[k,:].reshape(1,5))[0],min_z,max_z)}')
  print('\n')

**Test of model_predict_z on random points**

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
                        z_predit = deprocess_x(model_predict_z.predict(x_preproc.reshape(1,5))[0],min_z,max_z) 

                        # compute the error:
                        z_errors[s,:] = z_real - z_predit 
                        s += 1
                                   
    return z_errors

tc = 10
t_c = np.linspace(0, tc, 1000)
z_errors = test_z(t_c)

hist(z_errors,3,2)

for i in range(z_errors.shape[1]): 
    hist_i(z_errors,3,2,i,False)

#7-Simulate trajectories

def traj(t,x=np.array([0,0,0,0,0])):
    x_glob_init = np.zeros((11,))
    if (x == np.array([0,0,0,0,0])).all():
        x1 = (2*np.pi) * np.random.random_sample() - np.pi
        x2 = 2 * np.random.random_sample() -1
        x3 = 2 * np.random.random_sample() -1
        x4 = 2 * np.random.random_sample() -1
        x5 = (1-0.1) * np.random.random_sample() + 0.1
        x_glob_init = np.array([x1,x2,x3,x4,x5,0,0,0,0,0,0])
    
    else:
      x_glob_init[:5] = x
    print(x_glob_init[:5])

    solu = odeint(f_glob, x_glob_init, t)
    z = solu[:,5:]
    x = solu[:,:5]

    T_x = np.zeros(z.shape)
    T_x_hat = np.zeros(z.shape)

    z_modified = z.copy()
    x_predicted = np.zeros(((x.shape[0],2)))
    liste_k_1 = np.zeros(z.shape)
    for i in range (x_predicted.shape[0]):
        x_i = x[i,:]
        z_i = z_modified[i,:]
        z_i = preprocess_x(z_i,min_z,max_z)
        x_predicted_i = model_predict_x.predict(z_i.reshape(1,6))[0]     
        x_predicted[i] = deprocess_x(x_predicted_i,min_x,max_x)# + np.array([2*np.pi*k1_choosed,0])

        #in order to compare z to T(x_real) and T(x_predicted:)
        p = np.array([x_predicted[i][0],x_predicted[i][1],x[i,2],x[i,3],x[i,4]])
        x_predi_prep_i = preprocess_x(p,min_x,max_x)
        T_x_hat_i = model_predict_z.predict(x_predi_prep_i.reshape(1,5))[0] 
        T_x_hat[i] = deprocess_x(T_x_hat_i,min_z,max_z)

        x_prep = preprocess_x(x_i,min_x,max_x)
        T_x_i = model_predict_z.predict(x_prep.reshape(1,5))[0] 
        T_x[i] = deprocess_x(T_x_i,min_z,max_z)
    return x,z,x_predicted,T_x,T_x_hat
 x_simu = np.array([ 0.33019075, -0.71512765 , 0.52135956 , 0.06146751,  0.10147915]) # we choose this x in order to stay in the domain that we fixed and on which we trained the NN
  
 t = np.linspace(0,40,200)
x,z,x_predit,T_x,T_x_hat = traj(t,x_simu)

#Plot z and x:
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
    
# the input:
T = 1/0.05076530612244898
Amp = 0.5557902677321038 #max(input)
w=2*np.pi/T
fi = 0.3538347020102439 #np.arccos(input[0]/max(input))
y = Amp*np.cos(w*t-fi)
plt.plot(t,y,label='y')
plt.legend()

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

def inpu(t):
  return Amp*np.cos(w*t-fi)

def dot_inpu(t):
  return -Amp*w*np.sin(w*t-fi)

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
