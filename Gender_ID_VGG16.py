# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:25:08 2019

Nate Jermain Gender Identification CNN
"""
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df=pd.read_csv('C:/Users/w10007346/Pictures/list_attr_celeba.csv')

df.head()
df.columns.values

# get image labels for males
male=df[df['Male']==1][0:20000][['image_id', 'Male']]

female=df[df['Male']==-1][0:20000][['image_id','Male']]


from sklearn.model_selection import train_test_split
m_train_X, m_test_X, train_y, test_y = train_test_split(male['image_id'],male['Male'], random_state = 0, test_size=.2)
f_train_X, f_test_X, train_y, test_y = train_test_split(female['image_id'],female['Male'], random_state = 0, test_size=.2)


origin_path='C:/Users/w10007346/Pictures/img_align_celeba/'
train_path='C:/Users/w10007346/Pictures/Celeb_sets/train/'
valid_path='C:/Users/w10007346/Pictures/Celeb_sets/valid/'
fm='female/'
ml='male/'

import os

for file in m_train_X:
    os.rename(origin_path+file, train_path+ml+file)

for file in m_test_X:
    os.rename(origin_path+file, valid_path+ml+file)
    
    
for file in f_train_X:
    os.rename(origin_path+file, train_path+fm+file)

for file in f_test_X:
    os.rename(origin_path+file, valid_path+fm+file)
    
    
######## Modeling ################################
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization


from keras import models
from keras import layers

num_classes=2

vgg=VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(178, 218, 3))
vgg.summary()

# Freeze the layers except the last 2 layers
for layer in vgg.layers[:-5]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg.layers:
    print(layer, layer.trainable)
    

# Create the model
model = models.Sequential()


# Add the vgg convolutional base model
model.add(vgg)
 
# Add new layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# use early stopping to optimally terminate training through callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# save best model automatically
import h5py
mc= ModelCheckpoint('C:/Users/w10007346/Dropbox/CNN/Gender ID/best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
cb_list=[es,mc]


from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/train',
        target_size=(178, 218),
        batch_size=12,
        class_mode='categorical')


validation_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/valid',
        target_size=(178, 218),
        batch_size=12,
        class_mode='categorical')


model.fit_generator(
        train_generator,
        epochs=20,
        steps_per_epoch=2667,
        validation_data=validation_generator,
        validation_steps=667, callbacks=cb_list)





# load a saved model
from keras.models import load_model
saved_model = load_model('best_model.h5')

