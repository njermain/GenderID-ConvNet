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

# get image labels for either gender
male=df[df['Male']==1][0:20000][['image_id', 'Male']]

female=df[df['Male']==-1][0:20000][['image_id','Male']]


from sklearn.model_selection import train_test_split
m_train_X, m_test_X, train_y, test_y = train_test_split(male['image_id'],male['Male'], random_state = 0, test_size=.2)
f_train_X, f_test_X, train_y, test_y = train_test_split(female['image_id'],female['Male'], random_state = 0, test_size=.2)


origin_path='C:/Users/w10007346/Pictures/img_align_celeba/'
train_path='C:/Users/w10007346/Pictures/Celeb_sets/train/'
valid_path='C:/Users/w10007346/Pictures/Celeb_sets/valid/'
test_path='C:/Users/w10007346/Pictures/Celeb_sets/test/'
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
    
# construct seperate test set
test_m=df[df['Male']==1][-500:]
test_m=test_m.loc[:,'image_id']
test_f=df[df['Male']==-1][-500:]
test_f=test_f.loc[:,'image_id']

test_path='C:/Users/w10007346/Pictures/Celeb_sets/test/'

for file in test_m:
    os.rename(origin_path+file, test_path+ml+file)

for file in test_f:
    os.rename(origin_path+file, test_path+fm+file)
    

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
model.add(layers.Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# use early stopping to optimally terminate training through callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# save best model automatically
import h5py
mc= ModelCheckpoint('C:/Users/w10007346/Dropbox/CNN/Gender ID/best_model_2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
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


####### Testing ################################

# load a saved model
from keras.models import load_model
import os
os.chdir('C:/Users/w10007346/Dropbox/CNN/Gender ID')
saved_model = load_model('best_model.h5')

# generate data for test set of images
test_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/test',
        target_size=(178, 218),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# obtain predicted activation values for the last dense layer
test_generator.reset()
pred=saved_model.predict_generator(test_generator, verbose=1, steps=1000)
# determine the maximum activation value for each sample
predicted_class_indices=np.argmax(pred,axis=1)

# label each predicted value to correct gender
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# format file names to simply male or female
filenames=test_generator.filenames
filenz=[0]
for i in range(0,len(filenames)):
    filenz.append(filenames[i].split('\\')[0])
filenz=filenz[1:]

# determine the test set accuracy
match=[]
for i in range(0,len(filenames)):
    match.append(filenz[i]==predictions[i])
match.count(True)/1000


results=pd.DataFrame({"Filename":filenz,"Predictions":predictions})

pd.Series(filenz).str.match(pd.Series(predictions))
results.to_csv("GenderID_test_results.csv",index=False)

# predict for pictures of me
   
test_generator = data_generator.flow_from_directory(
        'C:/Users/w10007346/Pictures/Celeb_sets/test-me',
        target_size=(178, 218),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# obtain predicted activation values for the last dense layer
test_generator.reset()
pred=saved_model.predict_generator(test_generator, verbose=1, steps=2)
# determine the maximum activation value for each sample
predicted_class_indices=np.argmax(pred,axis=1)
    
    
    
    
    
    
me=read_and_prep_images(img_paths)
preds=model.predict(me)




