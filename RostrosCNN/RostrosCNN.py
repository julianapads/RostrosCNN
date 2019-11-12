# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:28:20 2019

@author: JuanMC
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
CatDog=Sequential()
# Convolucion
CatDog.add(Conv2D(32,(5,5), input_shape=(64,64,3), activation='relu'))
CatDog.add(MaxPooling2D(pool_size=(2,2)))
# Convolucion
CatDog.add(Conv2D(32,(5,5), input_shape=(64,64,3), activation='relu'))
CatDog.add(MaxPooling2D(pool_size=(2,2)))
CatDog.add(Flatten())
CatDog.add(Dense( units=512, activation='relu'  ))
CatDog.add(Dense( units=512, activation='relu'  ))
CatDog.add(Dense( units=1, activation='sigmoid'))
# parametros de enrtrenamiento
CatDog.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('C:/Users/julia/Desktop/training_set_short', target_size=(64,64), batch_size=32, class_mode='binary')
test_set=test_datagen.flow_from_directory('C:/Users/julia/Desktop/test_set_short', target_size=(64,64), batch_size=32, class_mode='binary')
CatDog.fit_generator(training_set, steps_per_epoch=38, epochs=150 , validation_data=test_set, validation_steps=3)
CatDog.save('Red_face5')
#int( np.ceil(1183/ 32) )
#int( np.ceil(60/ 32) )
