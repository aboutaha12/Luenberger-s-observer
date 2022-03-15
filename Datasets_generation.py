


#2-Creation of the datasets

inits = np.zeros((20**5,11)) # array of the 20**5 points obtained with the integration backward of (1) during tc
finals = np.zeros((20**5,11)) # array of the 20**5 points with the right couples (x,T(x))
# uniform distributions
x1_boucle = np.linspace(-np.pi, np.pi, 20)
x2_boucle = np.linspace(-2, 2, 20)
x3_boucle = np.linspace(-1, 1, 20)
x4_boucle = np.linspace(-1, 1, 20)
x5_boucle = np.linspace(0.1, 1, 20)

# points obtained by integrating backward (1) during tc
s = 0 # counter 
for i_x1 in x1_boucle:
    for i_x2 in x2_boucle:
        for i_x3 in x3_boucle:
            for i_x4 in x4_boucle:
                for i_x5 in x5_boucle:
                    x_init = np.array([i_x1, i_x2, i_x3, i_x4, i_x5])
                    # Integer (1) backward during tc:
                    sol = odeint(f_back, x_init, t)
                    x_init_back = sol[-1,:] # we take the last point 
                    # add the point to the array:
                    inits[s,:5] = x_init_back
                    s += 1

# np.save("dataset_init.npy", inits)

'''inits = np.load("dataset_init.npy")'''

# form the couples (x,T(x))
for i in range(20**5):
    #Integer (1) et (2) forward during tc
    solu = odeint(f_glob, inits[i,:], t) #inits[i,:]= [x_init_back,0,0,0,0,0,0]
    point = solu[-1,:]
    # Add the point to the array:
    finals[i,:] = point

# save the array:
#np.save("dataset_final.npy", finals) # This is the final dataset with the couples (x,T(x))


#finals = np.load("dataset_final.npy")

#3-Train and test sets

#finals = np.load("dataset_final.npy")
x = finals[:,:5]
z = finals[:,5:]

# split into train and test sets
x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.20, random_state=42)
'''np.save("x_train.npy", x_train)
np.save("z_train.npy", z_train)
np.save("x_test.npy", x_test)
np.save("z_test.npy", z_test)'''

'''x_train = np.load("x_train.npy")
z_train =np.load("z_train.npy")
x_test =np.load("x_test.npy")
z_test =np.load("z_test.npy")'''

#4-Preprocess the data

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

np.save("min_x_train",min_x)
np.save("max_x_train",max_x)
np.save("min_z_train",min_z)
np.save("max_z_train",max_z)

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


np.save("x_train_preproc.npy", x_train_preprocessed)
np.save("x_test_preproc.npy", x_test_preprocessed)
np.save("z_train_preproc.npy", z_train_preprocessed)
np.save("z_test_preproc.npy", z_test_preprocessed) 
