"""
Created on Tue Oct 29 09:54:11 2019

@author: Grupo 7
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
import tensorflow as tf
newModel = tf.keras.models.load_model('Red_face6')  
newModel.summary()
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('C:/Users/julia/Desktop/training_set_short', target_size=(128,128), batch_size=32, class_mode='binary')
test_set=test_datagen.flow_from_directory('C:/Users/julia/Desktop/test_set_short', target_size=(128,128), batch_size=32, class_mode='binary')
newModel.fit_generator(training_set, steps_per_epoch=38, epochs=3 , validation_data=test_set, validation_steps=3)
newModel.save('Red_face6')
int( np.ceil(1196/ 32) )
int( np.ceil(77/ 32) )

