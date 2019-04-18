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
import pandas as pd
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

incep=InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_shape=(178,218,3))
incep.summary()


num_classes = 2

my_new_model = Sequential()
my_new_model.add(incep)
my_new_model.add(Dense(128, activation='relu'))
my_new_model.add(BatchNormalization())
my_new_model.add(Dense(num_classes, activation='softmax'))

# Check the trainable status of the individual layers
for layer in my_new_model.layers:
    print(layer, layer.trainable)
  
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)


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


my_new_model.fit_generator(
        train_generator,
        epochs=20,
        steps_per_epoch=2667,
        validation_data=validation_generator,
        validation_steps=667)