# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:10:59 2023

@author: nandana s nair
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np  
import tensorflow as tf
from tensorflow.keras import models,layers


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator ( rescale=1./255,
                                    horizontal_flip=True,
                                    zoom_range=0.3,
                                    shear_range = 0.2
                                    )
train_dataset = train_datagen.flow_from_directory(
            'D:/project/Code_Malicious_images/dataset2/train',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=40)


#validation data
validation_dataset = train_datagen.flow_from_directory(
            'D:/project/Code_Malicious_images/dataset2/test',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=40)


#test data
test_dataset =train_datagen.flow_from_directory(
            'D:/project/Code_Malicious_images/dataset2/val',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=1)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 2

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_dataset),
    batch_size=32,
    validation_data=validation_dataset,
    validation_steps=len(validation_dataset),
    epochs=20,
)

model.save("mal_img_model.h5")