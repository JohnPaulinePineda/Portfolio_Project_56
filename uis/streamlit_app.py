##################################
# Loading Python Libraries
##################################
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from glob import glob
import cv2
import random
import math

import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, AveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img

##################################
# Defining file paths
##################################
DATASETS_FINAL_TEST_PATH = r"datasets\Brain_Tumor_MRI_Dataset\Testing"

##################################
# Defining the image category levels
# for the training data
##################################
diagnosis_code_dictionary_test = {'Te-no': 0,
                                  'Te-noTr': 0,
                                  'Te-gl': 1,
                                  'Te-glTr': 1,
                                  'Tr-me': 2,
                                  'Te-meTr': 2,
                                  'Te-pi': 3,
                                  'Te-piTr': 3}

##################################
# Defining the image category descriptions
# for the training data
##################################
diagnosis_description_dictionary_test = {'Te-no': 'No Tumor',
                                         'Te-noTr': 'No Tumor',
                                         'Te-gl': 'Glioma',
                                         'Te-glTr': 'Glioma',
                                         'Te-me': 'Meningioma',
                                         'Te-meTr': 'Meningioma',
                                         'Te-pi': 'Pituitary',
                                         'Te-piTr': 'Pituitary'}

##################################
# Consolidating the image path
# for the testing data
##################################
imageid_path_dictionary_test = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join("..", DATASETS_FINAL_TEST_PATH, '*','*.jpg'))}

##################################
# Consolidating the information
# from the testing datas
# into a dataframe
##################################
mri_images_test = pd.DataFrame.from_dict(imageid_path_dictionary_test, orient = 'index').reset_index()
mri_images_test.columns = ['Image_ID','Path']
classes = mri_images_test.Image_ID.str.split('_').str[0]
mri_images_test['Diagnosis'] = classes
mri_images_test['Target'] = mri_images_test['Diagnosis'].map(diagnosis_code_dictionary_test.get) 
mri_images_test['Class'] = mri_images_test['Diagnosis'].map(diagnosis_description_dictionary_test.get)

##################################
# Sampling a single image
# from the testing data
##################################
samples, features = mri_images_test.shape
plt.figure()
pic_id = random.randrange(0, samples)
picture = mri_images_test['Path'][pic_id]
image = cv2.imread(picture)

##################################
# Plotting the sampled image
# from the testing data
##################################

##################################
# Plotting using subplots
##################################
plt.figure(figsize=(15, 5))

##################################
# Resizing the image
##################################
resized_image = cv2.resize(image, (227, 227))

##################################
# Formulating the original image
##################################
plt.subplot(1, 4, 1)
plt.imshow(resized_image)
plt.title('Original Image', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the red channel
##################################
red_channel = np.zeros_like(resized_image)
red_channel[:, :, 0] = resized_image[:, :, 0]
plt.subplot(1, 4, 2)
plt.imshow(red_channel)
plt.title('Red Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the green channel
##################################
green_channel = np.zeros_like(resized_image)
green_channel[:, :, 1] = resized_image[:, :, 1]
plt.subplot(1, 4, 3)
plt.imshow(green_channel)
plt.title('Green Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the blue channel
##################################
blue_channel = np.zeros_like(resized_image)
blue_channel[:, :, 2] = resized_image[:, :, 2]
plt.subplot(1, 4, 4)
plt.imshow(blue_channel)
plt.title('Blue Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Consolidating all images
##################################
plt.show()
