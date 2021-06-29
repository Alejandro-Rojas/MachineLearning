# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:05:14 2021

@author: Alejandro
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import os
from tensorflow.keras import optimizers

def AlexNet(width, height, depth, classes, reg = 0.001):

    model = Sequential()
    inputShape = (height, width, depth)
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=inputShape, kernel_size=(11,11),strides=(4,4), padding='valid', kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid',kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid',kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid',kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid',kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096,kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(4096,kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(1000,kernel_regularizer= l2(reg)))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.5))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(classes,kernel_regularizer= l2(reg)))
    model.add(Activation('softmax'))
		# return the constructed network architecture
    return model
 


data_dir = os.path.abspath('Final_Training/')
train_dir = os.path.join(data_dir, "Images")
test_dir = os.path.join(data_dir, "Images")
os.path.exists(data_dir)

batch_size = 128
datagen_train = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(227,227),
                                                    shuffle = True,
                                                    class_mode = 'categorical')

generator_test = datagen_test.flow_from_directory(  directory=test_dir,
                                                    batch_size=batch_size,
                                                    target_size=(227,227),
                                                    class_mode = 'categorical',
                                                    shuffle = False)

steps_test = generator_test.n // batch_size
print(steps_test)
print(generator_test.n)

epochs = 50
steps_per_epoch = generator_train.n // batch_size
print(steps_per_epoch)

model = AlexNet(227,227,3,17)
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Adam = keras.optimizers.adam(lr=0.001, amsgrad = True)
#model.compile(loss='categorical_crossentropy', optimizer= Adam,\
#metrics=['accuracy'])

history= model.fit_generator(generator_train,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_test,
                           validation_steps = steps_test)

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)