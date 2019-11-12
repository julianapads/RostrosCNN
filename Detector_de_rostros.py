# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 21:12:47 2019

@author: Usuario
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import cv2
import os

from skimage.io import imread_collection

img_dir = 'Images/*' 
imgs = imread_collection(img_dir)#guarda todas las imagenes de la carpeta test 
#imagenes para entrenar
Entrada = []
Array_save = []
Load_Model='Red_face6'
if (Load_Model=='Red_face5.1'):r=0.06;n=64
else:r=0.05;n=128
"""------------------------   CUADROS   ---------------------------------------------"""


def Obtener_kernel(y, x, kernel):

        #Guardar los datos
    for i in range(n):
        for j in range(n):
            xx = x + i
            yy = y + j
            kernel[j,i] = imagen[yy ,xx]
            
    return kernel

def crear_circulos(coords, img, radio):
    
    cv2.circle(img,(coords[0],coords[1]),radio,255,5)
    start_point = (coords[0]-32,coords[1]-32)
    end_point = (coords[0]+32,coords[1]+32)
    cv2.rectangle(img, start_point, end_point,255, 5)
#    circle = plt.Circle(coords,.2,color='r')
#    img.add

while(1):
    print('BIENVENIDO \n Existen ',len(imgs)-1, ' imagenes para probar la neurona.\n')
    msj = str("Digite el numero de la imagen para usar de entrada, valores entre 0 - "+str(len(imgs)-1)+": ")
    opc = int(input(msj))
    print("Escogiste la opción: ", opc)
    plt.imshow(imgs[opc])
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    confirmacion = input("¿Desea escoger esta imagen? (y/n): ")
    global opcion
    if (confirmacion == 'y'):
        opcion = opc
        break
    else:
        os.system('cls')
        
imagen = imgs[opc]
y, x, _ = imagen.shape

newModel = tf.keras.models.load_model(Load_Model)  
#newModel.summary()


kernel = np.zeros((n,n,3)) # toma de datos

Puntos_encontrados = []

coory, coorx = 0,0

#for coory in range(y-(n-1)):
while coory <= y-(n):
#    for coorx in range(x-(n-1)):    
    while coorx <= x-(n):
#        if coorx > x-(n) : 
#            break
        Entrada = Obtener_kernel(coory, coorx, kernel) 
#        plt.imshow(Entrada/255)
#        plt.show()
        ImageP=np.expand_dims(Entrada, axis=0)
        Rostro=newModel.predict(ImageP)
#        print(Rostro)
        
        
        if Rostro[0][0]==1:
            Puntos_encontrados.append( [coorx+int(n/2) , coory+int(n/2)])
#            print('rostro')
            
    
        coorx += int(x*r)
    coorx = 0
    coory += int(y*r)
#    if (len(Puntos_encontrados))>10:
#        break


for coords in Puntos_encontrados:
    crear_circulos(coords,imagen,int(min(x,y)*0.018))

plt.imshow(imagen)
plt.tight_layout()
plt.show()