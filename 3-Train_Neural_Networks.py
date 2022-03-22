import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import EarlyStopping

x_train_preprocessed = np.load("x_train_preproc.npy")
x_test_preprocessed = np.load("x_test_preproc.npy")
z_train_preprocessed = np.load("z_train_preproc.npy")
z_test_preprocessed = np.load("z_test_preproc.npy")
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

train1()
train2()
