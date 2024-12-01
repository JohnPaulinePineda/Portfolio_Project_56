***
# Model Deployment : Classifying Brain Tumors from Magnetic Resonance Images by Leveraging Convolutional Neural Network-Based Multilevel Feature Extraction and Hierarchical Representation 

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *December 14, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Pre-Modelling Data Preparation](#1.6.1)
        * [1.6.2 Convolutional Neural Network Sequential Layer Development](#1.6.2)
            * [1.6.2.1 CNN With No Regularization](#1.6.2.1)
            * [1.6.2.2 CNN With Dropout Regularization](#1.6.2.2)
            * [1.6.2.3 CNN With Batch Normalization Regularization](#1.6.2.3)
            * [1.6.2.4 CNN With Dropout and Batch Normalization Regularization](#1.6.2.4)
        * [1.6.3 CNN With No Regularization Model Fitting | Hyperparameter Tuning | Validation](#1.6.3)
        * [1.6.4 CNN With Dropout Regularization Model Fitting | Hyperparameter Tuning | Validation](#1.6.4)
        * [1.6.5 CNN With Batch Normalization Regularization Model Fitting | Hyperparameter Tuning | Validation](#1.6.5)
        * [1.6.6 CNN With Dropout and Batch Normalization Regularization Model Fitting | Hyperparameter Tuning | Validation](#1.6.6)
        * [1.6.7 Model Selection](#1.6.7)
        * [1.6.8 Model Testing](#1.6.8)
        * [1.6.9 Model Inference](#1.6.9)
    * [1.7 Predictive Model Deployment Using Streamlit and Streamlit Community Cloud](#1.7)
        * [1.7.1 Model Application Programming Interface Code Development](#1.7.1)
        * [1.7.2 User Interface Application Code Development](#1.7.2)
        * [1.7.3 Web Application](#1.7.3)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

## 1.1 Data Background <a class="anchor" id="1.1"></a>

## 1.2 Data Description <a class="anchor" id="1.2"></a>


```python
##################################
# Loading Python Libraries
##################################

##################################
# Data Loading, Data Preprocessing
# and Exploratory Data Analysis
##################################
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
%matplotlib inline

import tensorflow as tf
import keras

from PIL import Image
from glob import glob
import cv2
import os
import random
import math

##################################
# Model Development
##################################
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model,load_model
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, AveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
from math import ceil

##################################
# Model Evaluation
##################################
from keras.metrics import PrecisionAtRecall, Recall 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

```


```python
##################################
# Setting random seed options
# for the analysis
##################################
def set_seed(seed=123):
    np.random.seed(seed) 
    tf.random.set_seed(seed) 
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

```


```python
##################################
# Filtering out unncessary warnings
##################################
import warnings
warnings.filterwarnings('ignore')

```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\Brain_Tumor_MRI_Dataset"
DATASETS_FINAL_TRAIN_PATH = r"datasets\Brain_Tumor_MRI_Dataset\Training"
DATASETS_FINAL_TEST_PATH = r"datasets\Brain_Tumor_MRI_Dataset\Testing"
MODELS_PATH = r"models"
PARAMETERS_PATH = r"parameters"
PIPELINES_PATH = r"pipelines"

```


```python
##################################
# Defining the image category levels
# for the training data
##################################
diagnosis_code_dictionary_train = {'Tr-no': 0,
                                   'Tr-noTr': 0,
                                   'Tr-gl': 1,
                                   'Tr-glTr': 1,
                                   'Tr-me': 2,
                                   'Tr-meTr': 2,
                                   'Tr-pi': 3,
                                   'Tr-piTr': 3}

##################################
# Defining the image category descriptions
# for the training data
##################################
diagnosis_description_dictionary_train = {'Tr-no': 'No Tumor',
                                          'Tr-noTr': 'No Tumor',
                                          'Tr-gl': 'Glioma',
                                          'Tr-glTr': 'Glioma',
                                          'Tr-me': 'Meningioma',
                                          'Tr-meTr': 'Meningioma',
                                          'Tr-pi': 'Pituitary',
                                          'Tr-piTr': 'Pituitary'}

```


```python
##################################
# Consolidating the image path
# for the training data
##################################
imageid_path_dictionary_train = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join("..", DATASETS_FINAL_TRAIN_PATH, '*','*.jpg'))}

```


```python
##################################
# Taking a snapshot of the dictionary
# for the training data
##################################
dict(list(imageid_path_dictionary_train.items())[0:5]) 

```




    {'Tr-glTr_0000': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Training\\glioma\\Tr-glTr_0000.jpg',
     'Tr-glTr_0001': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Training\\glioma\\Tr-glTr_0001.jpg',
     'Tr-glTr_0002': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Training\\glioma\\Tr-glTr_0002.jpg',
     'Tr-glTr_0003': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Training\\glioma\\Tr-glTr_0003.jpg',
     'Tr-glTr_0004': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Training\\glioma\\Tr-glTr_0004.jpg'}




```python
##################################
# Consolidating the information
# from the training datas
# into a dataframe
##################################
mri_images_train = pd.DataFrame.from_dict(imageid_path_dictionary_train, orient = 'index').reset_index()
mri_images_train.columns = ['Image_ID','Path']
classes = mri_images_train.Image_ID.str.split('_').str[0]
mri_images_train['Diagnosis'] = classes
mri_images_train['Target'] = mri_images_train['Diagnosis'].map(diagnosis_code_dictionary_train.get) 
mri_images_train['Class'] = mri_images_train['Diagnosis'].map(diagnosis_description_dictionary_train.get) 

```


```python
##################################
# Performing a general exploration of the training data
##################################
print('Dataset Dimensions: ')
display(mri_images_train.shape)

```

    Dataset Dimensions: 
    


    (5712, 5)



```python
##################################
# Listing the column names and data types
# for the training data
##################################
print('Column Names and Data Types:')
display(mri_images_train.dtypes)

```

    Column Names and Data Types:
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
    dtype: object



```python
##################################
# Taking a snapshot of the training data
##################################
mri_images_train.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tr-glTr_0000</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tr-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tr-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tr-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tr-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration of the numeric variables
# for the training data
##################################
print('Numeric Variable Summary:')
display(mri_images_train.describe(include='number').transpose())

```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Target</th>
      <td>5712.0</td>
      <td>1.465336</td>
      <td>1.147892</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variables
# for the training data
##################################
print('Object Variable Summary:')
display(mri_images_train.describe(include='object').transpose())

```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Image_ID</th>
      <td>5712</td>
      <td>5712</td>
      <td>Tr-pi_1440</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Path</th>
      <td>5712</td>
      <td>5712</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\p...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Diagnosis</th>
      <td>5712</td>
      <td>8</td>
      <td>Tr-no</td>
      <td>1585</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>5712</td>
      <td>4</td>
      <td>No Tumor</td>
      <td>1595</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the target variable
# for the training data
##################################
mri_images_train.Class.value_counts()

```




    Class
    No Tumor      1595
    Pituitary     1457
    Meningioma    1339
    Glioma        1321
    Name: count, dtype: int64




```python
##################################
# Performing a general exploration of the target variable
# for the training data
##################################
mri_images_train.Class.value_counts(normalize=True)

```




    Class
    No Tumor      0.279237
    Pituitary     0.255077
    Meningioma    0.234419
    Glioma        0.231268
    Name: proportion, dtype: float64




```python
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
```


```python
##################################
# Consolidating the image path
# for the testing data
##################################
imageid_path_dictionary_test = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join("..", DATASETS_FINAL_TEST_PATH, '*','*.jpg'))}

```


```python
##################################
# Taking a snapshot of the dictionary
# for the testing data
##################################
dict(list(imageid_path_dictionary_test.items())[0:5])

```




    {'Te-glTr_0000': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Testing\\glioma\\Te-glTr_0000.jpg',
     'Te-glTr_0001': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Testing\\glioma\\Te-glTr_0001.jpg',
     'Te-glTr_0002': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Testing\\glioma\\Te-glTr_0002.jpg',
     'Te-glTr_0003': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Testing\\glioma\\Te-glTr_0003.jpg',
     'Te-glTr_0004': '..\\datasets\\Brain_Tumor_MRI_Dataset\\Testing\\glioma\\Te-glTr_0004.jpg'}




```python
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

```


```python
##################################
# Performing a general exploration of the testing data
##################################
print('Dataset Dimensions: ')
display(mri_images_test.shape)

```

    Dataset Dimensions: 
    


    (1311, 5)



```python
##################################
# Listing the column names and data types
# for the testing data
##################################
print('Column Names and Data Types:')
display(mri_images_test.dtypes)

```

    Column Names and Data Types:
    


    Image_ID      object
    Path          object
    Diagnosis     object
    Target       float64
    Class         object
    dtype: object



```python
##################################
# Taking a snapshot of the testing data
##################################
mri_images_test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Te-glTr_0000</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Te-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Te-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Te-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Te-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration of the numeric variables
# for the testing data
##################################
print('Numeric Variable Summary:')
display(mri_images_test.describe(include='number').transpose())

```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Target</th>
      <td>1015.0</td>
      <td>1.20197</td>
      <td>1.245741</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variables
# for the testing data
##################################
print('Object Variable Summary:')
display(mri_images_test.describe(include='object').transpose())

```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Image_ID</th>
      <td>1311</td>
      <td>1311</td>
      <td>Te-pi_0299</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Path</th>
      <td>1311</td>
      <td>1311</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\pi...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Diagnosis</th>
      <td>1311</td>
      <td>8</td>
      <td>Te-no</td>
      <td>395</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>1311</td>
      <td>4</td>
      <td>No Tumor</td>
      <td>405</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the target variable
# for the testing data
##################################
mri_images_test.Class.value_counts()

```




    Class
    No Tumor      405
    Meningioma    306
    Glioma        300
    Pituitary     300
    Name: count, dtype: int64




```python
##################################
# Performing a general exploration of the target variable
# for the testing data
##################################
mri_images_test.Class.value_counts(normalize=True)

```




    Class
    No Tumor      0.308924
    Meningioma    0.233410
    Glioma        0.228833
    Pituitary     0.228833
    Name: proportion, dtype: float64



## 1.3 Data Quality Assessment <a class="anchor" id="1.3"></a>


```python
##################################
# Counting the number of duplicated images
# for the training data
##################################
mri_images_train.duplicated().sum()

```




    np.int64(0)




```python
##################################
# Gathering the number of null images
##################################
mri_images_train.isnull().sum()

```




    Image_ID     0
    Path         0
    Diagnosis    0
    Target       0
    Class        0
    dtype: int64




```python
##################################
# Counting the number of duplicated images
# for the testing data
##################################
mri_images_test.duplicated().sum()

```




    np.int64(0)




```python
##################################
# Gathering the number of null images
##################################
mri_images_test.isnull().sum()

```




    Image_ID       0
    Path           0
    Diagnosis      0
    Target       296
    Class          0
    dtype: int64



## 1.4 Data Preprocessing <a class="anchor" id="1.4"></a>


```python
##################################
# Including the pixel information
# of the actual images in array format
# for the training data
# into a dataframe
##################################
mri_images_train['Image'] = mri_images_train['Path'].map(lambda x: np.asarray(Image.open(x).resize((200,200))))

```


```python
##################################
# Listing the column names and data types
# for the training data
##################################
print('Column Names and Data Types:')
display(mri_images_train.dtypes)

```

    Column Names and Data Types:
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
    Image        object
    dtype: object



```python
##################################
# Taking a snapshot of the training data
##################################
mri_images_train.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tr-glTr_0000</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tr-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tr-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tr-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tr-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>Tr-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking a snapshot of the training data
##################################
n_samples = 5
fig, m_axs = plt.subplots(4, n_samples, figsize = (2*n_samples, 10))
for n_axs, (type_name, type_rows) in zip(m_axs, mri_images_train.sort_values(['Class']).groupby('Class')):
    n_axs[2].set_title(type_name, fontsize = 14, weight = 'bold')
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=123).iterrows()):       
        picture = c_row['Path']
        image = cv2.imread(picture)
        resized_image = cv2.resize(image, (500,500))
        c_ax.imshow(resized_image)
        c_ax.axis('off')
        
```


    
![png](output_41_0.png)
    



```python
##################################
# Sampling a single image
# from the training data
##################################
samples, features = mri_images_train.shape
plt.figure()
pic_id = random.randrange(0, samples)
picture = mri_images_train['Path'][pic_id]
image = cv2.imread(picture) 

```


    <Figure size 640x480 with 0 Axes>



```python
##################################
# Plotting using subplots
##################################
plt.figure(figsize=(15, 5))

##################################
# Formulating the original image
##################################
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the blue channel
##################################
plt.subplot(1, 4, 2)
plt.imshow(image[ : , : , 0])
plt.title('Blue Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the green channel
##################################
plt.subplot(1, 4, 3)
plt.imshow(image[ : , : , 1])
plt.title('Green Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the red channel
##################################
plt.subplot(1, 4, 4)
plt.imshow(image[ : , : , 2])
plt.title('Red Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Consolidating all images
##################################
plt.show()

```


    
![png](output_43_0.png)
    



```python
##################################
# Determining the image shape
##################################
print('Image Shape:')
display(image.shape)

```

    Image Shape:
    


    (512, 512, 3)



```python
##################################
# Determining the image height
##################################
print('Image Height:')
display(image.shape[0])

```

    Image Height:
    


    512



```python
##################################
# Determining the image width
##################################
print('Image Width:')
display(image.shape[1])

```

    Image Width:
    


    512



```python
##################################
# Determining the image dimension
##################################
print('Image Dimension:')
display(image.ndim)

```

    Image Dimension:
    


    3



```python
##################################
# Determining the image size
##################################
print('Image Size:')
display(image.size)

```

    Image Size:
    


    786432



```python
##################################
# Determining the image data type
##################################
print('Image Data Type:')
display(image.dtype)

```

    Image Data Type:
    


    dtype('uint8')



```python
##################################
# Determining the maximum RGB value
##################################
print('Image Maximum RGB:')
display(image.max())

```

    Image Maximum RGB:
    


    np.uint8(255)



```python
##################################
# Determining the minimum RGB value
##################################
print('Image Minimum RGB:')
display(image.min())

```

    Image Minimum RGB:
    


    np.uint8(0)



```python
##################################
# Identifying the path for the images
# and defining image categories 
##################################
path_train = (os.path.join("..", DATASETS_FINAL_TRAIN_PATH))
classes=["notumor", "glioma", "meningioma", "pituitary"]
num_classes = len(classes)
batch_size = 32

```


```python
##################################
# Creating subsets of images
# for model training and
# setting the parameters for
# real-time data augmentation
# at each epoch
##################################
set_seed()
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=2,
                                   width_shift_range=0.02,
                                   height_shift_range=0.02,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   shear_range=0.02,
                                   zoom_range=0.02,
                                   validation_split=0.2)

##################################
# Loading the model training images
##################################
train_gen = train_datagen.flow_from_directory(directory=path_train, 
                                              target_size=(227, 227),
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True, 
                                              classes=classes,
                                              batch_size=batch_size, 
                                              color_mode="grayscale")

```

    Found 4571 images belonging to 4 classes.
    


```python
##################################
# Loading samples of augmented images
# for the training set
##################################
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    batch = next(train_gen)
    images, labels = batch
    axes[i].imshow(images[0]) 
    axes[i].set_title(f"Label: {labels[0]}")
    axes[i].axis('off')
plt.show()

```


    
![png](output_54_0.png)
    



```python
##################################
# Creating subsets of images
# for model validation and
# setting the parameters for
# real-time data augmentation
# at each epoch
##################################
set_seed()
val_datagen = ImageDataGenerator(rescale=1./255, 
                                 validation_split=0.2)

##################################
# Loading the model evaluation images
##################################
val_gen = val_datagen.flow_from_directory(directory=path_train, 
                                            target_size=(227, 227),
                                            class_mode='categorical',
                                            subset='validation',
                                            shuffle=False, 
                                            classes=classes,
                                            batch_size=batch_size, 
                                            color_mode="grayscale")

```

    Found 1141 images belonging to 4 classes.
    


```python
##################################
# Loading samples of original images
# for the validation set
##################################
images, labels = next(val_gen)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(range(0, 5)):
    axes[i].imshow(images[idx])
    axes[i].set_title(f"Label: {labels[0]}")
    axes[i].axis('off')
plt.show()

```


    
![png](output_56_0.png)
    



```python
##################################
# Including the pixel information
# of the actual images in array format
# for the testing data
# into a dataframe
##################################
mri_images_test['Image'] = mri_images_test['Path'].map(lambda x: np.asarray(Image.open(x).resize((200,200))))

```


```python
##################################
# Listing the column names and data types
# for the testing data
##################################
print('Column Names and Data Types:')
display(mri_images_test.dtypes)

```

    Column Names and Data Types:
    


    Image_ID      object
    Path          object
    Diagnosis     object
    Target       float64
    Class         object
    Image         object
    dtype: object



```python
##################################
# Taking a snapshot of the testing data
##################################
mri_images_test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Te-glTr_0000</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Te-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Te-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Te-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Te-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1.0</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking a snapshot of the testing data
##################################
n_samples = 5
fig, m_axs = plt.subplots(4, n_samples, figsize = (2*n_samples, 10))
for n_axs, (type_name, type_rows) in zip(m_axs, mri_images_test.sort_values(['Class']).groupby('Class')):
    n_axs[2].set_title(type_name, fontsize = 14, weight = 'bold')
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=123).iterrows()):       
        picture = c_row['Path']
        image = cv2.imread(picture)
        resized_image = cv2.resize(image, (500,500))
        c_ax.imshow(resized_image)
        c_ax.axis('off')

```


    
![png](output_60_0.png)
    



```python
##################################
# Sampling a single image
# from the testing data
##################################
samples, features = mri_images_test.shape
plt.figure()
pic_id = random.randrange(0, samples)
picture = mri_images_test['Path'][pic_id]
image = cv2.imread(picture)

```


    <Figure size 640x480 with 0 Axes>



```python
##################################
# Plotting using subplots
##################################
plt.figure(figsize=(15, 5))

##################################
# Formulating the original image
##################################
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the blue channel
##################################
plt.subplot(1, 4, 2)
plt.imshow(image[ : , : , 0])
plt.title('Blue Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the green channel
##################################
plt.subplot(1, 4, 3)
plt.imshow(image[ : , : , 1])
plt.title('Green Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the red channel
##################################
plt.subplot(1, 4, 4)
plt.imshow(image[ : , : , 2])
plt.title('Red Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Consolidating all images
##################################
plt.show()

```


    
![png](output_62_0.png)
    



```python
##################################
# Determining the image shape
##################################
print('Image Shape:')
display(image.shape)

```

    Image Shape:
    


    (512, 512, 3)



```python
##################################
# Determining the image height
##################################
print('Image Height:')
display(image.shape[0])

```

    Image Height:
    


    512



```python
##################################
# Determining the image width
##################################
print('Image Width:')
display(image.shape[1])

```

    Image Width:
    


    512



```python
##################################
# Determining the image dimension
##################################
print('Image Dimension:')
display(image.ndim)

```

    Image Dimension:
    


    3



```python
##################################
# Determining the image size
##################################
print('Image Size:')
display(image.size)

```

    Image Size:
    


    786432



```python
##################################
# Determining the image data type
##################################
print('Image Data Type:')
display(image.dtype)

```

    Image Data Type:
    


    dtype('uint8')



```python
##################################
# Determining the maximum RGB value
##################################
print('Image Maximum RGB:')
display(image.max())

```

    Image Maximum RGB:
    


    np.uint8(255)



```python
##################################
# Determining the minimum RGB value
##################################
print('Image Minimum RGB:')
display(image.min())

```

    Image Minimum RGB:
    


    np.uint8(0)



```python
##################################
# Identifying the path for the images
# and defining image categories 
##################################
path_test = (os.path.join("..", DATASETS_FINAL_TEST_PATH))
classes=["notumor", "glioma", "meningioma", "pituitary"]
num_classes = len(classes)
batch_size = 32

```


```python
##################################
# Creating subsets of images
# for model testing and
# setting the parameters for
# real-time data augmentation
# at each epoch
##################################
set_seed()
test_datagen = ImageDataGenerator(rescale=1./255)

##################################
# Loading the model testing images
##################################
test_gen = test_datagen.flow_from_directory(directory=path_test, 
                                            target_size=(227, 227),
                                            class_mode='categorical',
                                            shuffle=False, 
                                            classes=classes,
                                            batch_size=batch_size, 
                                            color_mode="grayscale")

```

    Found 1311 images belonging to 4 classes.
    


```python
##################################
# Loading samples of augmented images
# for the testing set
##################################
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    batch = next(test_gen)
    images, labels = batch
    axes[i].imshow(images[0]) 
    axes[i].set_title(f"Label: {labels[0]}")
    axes[i].axis('off')
plt.show()

```


    
![png](output_73_0.png)
    


## 1.5 Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>


```python
##################################
# Consolidating summary statistics
# for the image pixel values
##################################
samples, features = mri_images_train.shape
mean_val = []
std_dev_val = []
max_val = []
min_val = []

for i in range(0, samples):
    mean_val.append(mri_images_train['Image'][i].mean())
    std_dev_val.append(np.std(mri_images_train['Image'][i]))
    max_val.append(mri_images_train['Image'][i].max())
    min_val.append(mri_images_train['Image'][i].min())

imageEDA = mri_images_train.loc[:,['Image', 'Class','Path']]
imageEDA['Mean'] = mean_val
imageEDA['StDev'] = std_dev_val
imageEDA['Max'] = max_val
imageEDA['Min'] = min_val

```


```python
##################################
# Consolidating the overall mean
# for the pixel intensity means
# grouped by categories
##################################
imageEDA.groupby(['Class'])['Mean'].mean()

```




    Class
    Glioma        32.716871
    Meningioma    43.487954
    No Tumor      60.815724
    Pituitary     49.273456
    Name: Mean, dtype: float64




```python
##################################
# Consolidating the overall minimum
# for the pixel intensity means
# grouped by categories
##################################
imageEDA.groupby(['Class'])['Mean'].min()

```




    Class
    Glioma        13.701850
    Meningioma    18.233400
    No Tumor       9.770775
    Pituitary     24.699575
    Name: Mean, dtype: float64




```python
##################################
# Consolidating the overall maximum
# for the pixel intensity means
# grouped by categories
##################################
imageEDA.groupby(['Class'])['Mean'].max()

```




    Class
    Glioma         68.372425
    Meningioma    137.765375
    No Tumor      125.066725
    Pituitary     102.007950
    Name: Mean, dtype: float64




```python
##################################
# Consolidating the overall standard deviation
# for the pixel intensity means
# grouped by categories
##################################
imageEDA.groupby(['Class'])['Mean'].std()

```




    Class
    Glioma         8.565834
    Meningioma    14.307165
    No Tumor      21.338225
    Pituitary      8.222902
    Name: Mean, dtype: float64




```python
##################################
# Formulating the mean distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'Mean', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Mean Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_81_0.png)
    



```python
##################################
# Formulating the maximum distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'Max', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Maximum Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_82_0.png)
    



```python
##################################
# Formulating the minimum distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'Min', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Minimum Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_83_0.png)
    



```python
##################################
# Formulating the standard deviation distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'StDev', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Standard Deviation Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_84_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# by category of the image pixel values
##################################
plt.figure(figsize=(10,6))
sns.set(style="ticks", font_scale = 1)
ax = sns.scatterplot(data=imageEDA, x="Mean", y=imageEDA['StDev'], hue='Class', alpha=0.5)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(rotation=0, fontsize = 12)
ax.set_xlabel('Image Pixel Intensity Mean',fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
plt.title('Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize = 14, weight='bold');

```


    
![png](output_85_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# by category of the image pixel values
##################################
scatterplot = sns.FacetGrid(imageEDA, col="Class", height=6)
scatterplot.map_dataframe(sns.scatterplot, x='Mean', y='StDev', alpha=0.5)
scatterplot.set_titles(col_template="{col_name}", row_template="{row_name}", size=18)
scatterplot.fig.subplots_adjust(top=.8)
scatterplot.fig.suptitle('Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold')
axes = scatterplot.axes.flatten()
axes[0].set_ylabel('Image Pixel Intensity Standard Deviation')
for ax in axes:
    ax.set_xlabel('Image Pixel Intensity Mean')
scatterplot.fig.tight_layout()

```


    
![png](output_86_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
##################################
def getImage(path):
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (300,300))
    return OffsetImage(resized_image, zoom = 0.1)

DF_sample = imageEDA.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(0,120)
plt.title('Overall: Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path in zip(DF_sample['Mean'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_87_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Glioma class
##################################
path_glioma = (os.path.join("..", DATASETS_FINAL_TRAIN_PATH,'glioma/'))
imageEDA_glioma = imageEDA.loc[imageEDA['Class'] == 'Glioma']

DF_sample = imageEDA_glioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('Glioma: Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_glioma in zip(DF_sample['Mean'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_glioma), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_88_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Viral Pneumonia class
##################################
path_meningioma = (os.path.join("..", DATASETS_FINAL_TRAIN_PATH,'meningioma/'))
imageEDA_meningioma = imageEDA.loc[imageEDA['Class'] == 'Meningioma']

DF_sample = imageEDA_meningioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('Meningioma: Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_meningioma in zip(DF_sample['Mean'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_meningioma), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_89_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Pituitary class
##################################
path_pituitary = (os.path.join("..", DATASETS_FINAL_TRAIN_PATH,'pituitary/'))
imageEDA_pituitary = imageEDA.loc[imageEDA['Class'] == 'Pituitary']

DF_sample = imageEDA_pituitary.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(0, 140)
ax.set_ylim(10,110)
plt.title('Pituitary: Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_pituitary in zip(DF_sample['Mean'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_pituitary), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_90_0.png)
    



```python
##################################
# Formulating the mean and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the No Tumor class
##################################
path_no_tumor = (os.path.join("..", DATASETS_FINAL_TRAIN_PATH,'notumor/'))
imageEDA_no_tumor = imageEDA.loc[imageEDA['Class'] == 'No Tumor']

DF_sample = imageEDA_no_tumor.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('No Tumor: Image Pixel Intensity Mean and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_no_tumor in zip(DF_sample['Mean'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_no_tumor), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_91_0.png)
    



```python
#################################
# Formulating the minimum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
##################################
DF_sample = imageEDA.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Min", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Minimum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(0,120)
plt.title('Overall: Image Pixel Intensity Minimum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path in zip(DF_sample['Min'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_92_0.png)
    



```python
##################################
# Formulating the minimum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Glioma class
##################################
DF_sample = imageEDA_glioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Min", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Minimum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('Glioma: Image Pixel Intensity Minimum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_glioma in zip(DF_sample['Min'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_glioma), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
```


    
![png](output_93_0.png)
    



```python
##################################
# Formulating the minimum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Meningioma class
##################################
DF_sample = imageEDA_meningioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Min", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Minimum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('Meningioma: Image Pixel Intensity Minimum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_meningioma in zip(DF_sample['Min'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_meningioma), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_94_0.png)
    



```python
##################################
# Formulating the minimum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Pituitary class
##################################
DF_sample = imageEDA_pituitary.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Min", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Minimum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('Pituitary: Image Pixel Intensity Minimum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_pituitary in zip(DF_sample['Min'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_pituitary), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_95_0.png)
    



```python
##################################
# Formulating the minimum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the No Tumor class
##################################
DF_sample = imageEDA_no_tumor.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Min", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Minimum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(-5,145)
ax.set_ylim(10,110)
plt.title('No Tumor: Image Pixel Intensity Minimum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_no_tumor in zip(DF_sample['Min'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_no_tumor), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_96_0.png)
    



```python
#################################
# Formulating the maximum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
##################################
DF_sample = imageEDA.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Max", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Maximum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(115,265)
ax.set_ylim(0,120)
plt.title('Overall: Image Pixel Intensity Maximum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path in zip(DF_sample['Max'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_97_0.png)
    



```python
##################################
# Formulating the maximum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Glioma class
##################################
DF_sample = imageEDA_glioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Max", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Maximum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(115,265)
ax.set_ylim(10,110)
plt.title('Glioma: Image Pixel Intensity Maximum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_glioma in zip(DF_sample['Max'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_glioma), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_98_0.png)
    



```python
##################################
# Formulating the maximum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Meningioma class
##################################
DF_sample = imageEDA_meningioma.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Max", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Maximum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(115,265)
ax.set_ylim(10,110)
plt.title('Meningioma: Image Pixel Intensity Maximum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_meningioma in zip(DF_sample['Max'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_meningioma), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_99_0.png)
    



```python
##################################
# Formulating the maximum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the Pituitary class
##################################
DF_sample = imageEDA_pituitary.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Max", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Maximum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(115,265)
ax.set_ylim(10,110)
plt.title('Pituitary: Image Pixel Intensity Maximum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_pituitary in zip(DF_sample['Max'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_pituitary), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_100_0.png)
    



```python
##################################
# Formulating the maximum and standard deviation 
# scatterplot distribution
# of the image pixel values
# represented as actual images
# for the No Tumor class
##################################
DF_sample = imageEDA_no_tumor.sample(frac=1.0, replace=False, random_state=123)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(15,9))
ab = sns.scatterplot(data=DF_sample, x="Max", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Pixel Intensity Maximum', fontsize=14, weight='bold')
ax.set_ylabel('Image Pixel Intensity Standard Deviation', fontsize=14, weight='bold')
ax.set_xlim(115,265)
ax.set_ylim(10,110)
plt.title('No Tumor: Image Pixel Intensity Maximum and Standard Deviation Distribution', fontsize=14, weight='bold');

for x0, y0, path_no_tumor in zip(DF_sample['Max'], DF_sample['StDev'], paths):
    ab = AnnotationBbox(getImage(path_no_tumor), (x0, y0), frameon=False)
    ax.add_artist(ab)

```


    
![png](output_101_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

## 1.6 Predictive Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Pre-Modelling Data Preparation <a class="anchor" id="1.6.1"></a>

### 1.6.2 Convolutional Neural Network Sequential Layer Development <a class="anchor" id="1.6.2"></a>


```python
##################################
# Defining a function for
# plotting the loss profile
# of the training and validation sets
#################################
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 8))
    
    # Plotting training and validation loss
    plt.subplot(2, 1, 1)  # First subplot for loss
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{model_name} Training and Validation Loss', fontsize=16, weight='bold', pad=20)
    plt.ylim(-0.2, 2.2)
    plt.yticks([x * 0.50 for x in range(0, 5)])
    plt.xlim(-1, 21)
    plt.xticks([x for x in range(0, 21)])
    plt.xlabel('Epoch', fontsize=14, weight='bold')
    plt.ylabel('Loss', fontsize=14, weight='bold')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Plotting training and validation recall
    plt.subplot(2, 1, 2)  # Second subplot for recall
    plt.plot(history.history['recall'], label='Train Recall', color='green')
    plt.plot(history.history['val_recall'], label='Validation Recall', color='red')
    plt.title(f'{model_name} Training and Validation Recall', fontsize=16, weight='bold', pad=20)
    plt.ylim(-0.1, 1.1) 
    plt.yticks([x * 0.25 for x in range(0, 5)])
    plt.xlim(-1, 21)
    plt.xticks([x for x in range(0, 21)])
    plt.xlabel('Epoch', fontsize=14, weight='bold')
    plt.ylabel('Recall', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Adjusting layout and show the plots
    plt.tight_layout(pad=2.0)
    plt.show()
    
```


```python
##################################
# Defining the model file paths
#################################
NR_SIMPLE_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "nr_simple_best_model.keras")
DR_SIMPLE_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "dr_simple_best_model.keras")
BNR_SIMPLE_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "bnr_simple_best_model.keras")
CDRBNR_SIMPLE_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "cdrbnr_simple_best_model.keras")

NR_COMPLEX_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "nr_complex_best_model.keras")
DR_COMPLEX_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "dr_complex_best_model.keras")
BNR_COMPLEX_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "bnr_complex_best_model.keras")
CDRBNR_COMPLEX_BEST_MODEL_PATH = os.path.join("..", MODELS_PATH, "cdrbnr_complex_best_model.keras")

```


```python
##################################
# Defining the model callback configuration
# for model training
#################################
early_stopping = EarlyStopping(
    monitor='val_loss',                      # Defining the metric to monitor
    patience=10,                             # Defining the number of epochs to wait before stopping if no improvement
    min_delta=1e-4 ,                         # Defining the minimum change in the monitored quantity to qualify as an improvement
    restore_best_weights=True                # Restoring the weights from the best epoch
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',                      # Defining the metric to monitor
    factor=0.1,                              # Reducing the learning rate by a factor of 10%
    patience=3,                              # Defining the number of epochs to wait before reducing learning rate
    min_lr=1e-6                              # Defining the lower bound on the learning rate
)

nr_simple_model_checkpoint = ModelCheckpoint(
    filepath=NR_SIMPLE_BEST_MODEL_PATH,      # Defining the file path for saving
    monitor='val_loss',                      # Defining the metric to monitor
    save_best_only=True,                     # Saving only the best model
    save_weights_only=False,                 # Saving the entire model, not just weights
)

dr_simple_model_checkpoint = ModelCheckpoint(
    filepath=DR_SIMPLE_BEST_MODEL_PATH,      # Defining the file path for saving
    monitor='val_loss',                      # Defining the metric to monitor
    save_best_only=True,                     # Saving only the best model
    save_weights_only=False,                 # Saving the entire model, not just weights
)

bnr_simple_model_checkpoint = ModelCheckpoint(
    filepath=BNR_SIMPLE_BEST_MODEL_PATH,     # Defining the file path for saving
    monitor='val_loss',                      # Defining the metric to monitor
    save_best_only=True,                     # Saving only the best model
    save_weights_only=False,                 # Saving the entire model, not just weights
)

cdrbnr_simple_model_checkpoint = ModelCheckpoint(
    filepath=CDRBNR_SIMPLE_BEST_MODEL_PATH,  # Defining the file path for saving
    monitor='val_loss',                      # Defining the metric to monitor
    save_best_only=True,                     # Saving only the best model
    save_weights_only=False,                 # Saving the entire model, not just weights
)


nr_complex_model_checkpoint = ModelCheckpoint(
    filepath=NR_COMPLEX_BEST_MODEL_PATH,      # Defining the file path for saving
    monitor='val_loss',                       # Defining the metric to monitor
    save_best_only=True,                      # Saving only the best model
    save_weights_only=False,                  # Saving the entire model, not just weights
)

dr_complex_model_checkpoint = ModelCheckpoint(
    filepath=DR_COMPLEX_BEST_MODEL_PATH,      # Defining the file path for saving
    monitor='val_loss',                       # Defining the metric to monitor
    save_best_only=True,                      # Saving only the best model
    save_weights_only=False,                  # Saving the entire model, not just weights
)

bnr_complex_model_checkpoint = ModelCheckpoint(
    filepath=BNR_COMPLEX_BEST_MODEL_PATH,     # Defining the file path for saving
    monitor='val_loss',                       # Defining the metric to monitor
    save_best_only=True,                      # Saving only the best model
    save_weights_only=False,                  # Saving the entire model, not just weights
)

cdrbnr_complex_model_checkpoint = ModelCheckpoint(
    filepath=CDRBNR_COMPLEX_BEST_MODEL_PATH,  # Defining the file path for saving
    monitor='val_loss',                       # Defining the metric to monitor
    save_best_only=True,                      # Saving only the best model
    save_weights_only=False,                  # Saving the entire model, not just weights
)
```

#### 1.6.2.1 CNN With No Regularization <a class="anchor" id="1.6.2.1"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with no regularization
##################################
set_seed()
batch_size = 32
model_nr_simple = Sequential(name="model_nr_simple")
model_nr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="nr_simple_conv2d_0"))
model_nr_simple.add(MaxPooling2D(pool_size=(2, 2), name="nr_simple_max_pooling2d_0"))
model_nr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_simple_conv2d_1"))
model_nr_simple.add(MaxPooling2D(pool_size=(2, 2), name="nr_simple_max_pooling2d_1"))
model_nr_simple.add(Flatten(name="nr_simple_flatten"))
model_nr_simple.add(Dense(units=32, activation='relu', name="nr_simple_dense_0"))
model_nr_simple.add(Dense(units=num_classes, activation='softmax', name="nr_simple_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a simple CNN with no regularization
##################################
print(model_nr_simple.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_nr_simple"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ nr_simple_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">80</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_max_pooling2d_0            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">1,168</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_max_pooling2d_1            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │       <span style="color: #00af00; text-decoration-color: #00af00">1,605,664</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_simple_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">132</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,044</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,044</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a simple CNN with no regularization
##################################
model_nr_simple_layer_names = [layer.name for layer in model_nr_simple.layers]
print("Layer Names:", model_nr_simple_layer_names)

```

    Layer Names: ['nr_simple_conv2d_0', 'nr_simple_max_pooling2d_0', 'nr_simple_conv2d_1', 'nr_simple_max_pooling2d_1', 'nr_simple_flatten', 'nr_simple_dense_0', 'nr_simple_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with no regularization
##################################
for layer in model_nr_simple.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")

```

    Layer: nr_simple_conv2d_0, Number of Weights: 2
    Layer: nr_simple_max_pooling2d_0, Number of Weights: 0
    Layer: nr_simple_conv2d_1, Number of Weights: 2
    Layer: nr_simple_max_pooling2d_1, Number of Weights: 0
    Layer: nr_simple_flatten, Number of Weights: 0
    Layer: nr_simple_dense_0, Number of Weights: 2
    Layer: nr_simple_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of parameters
# for each model layer
# for a simple CNN with no regularization
##################################
total_parameters = 0
for layer in model_nr_simple.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: nr_simple_conv2d_0, Parameters: 80
    Layer: nr_simple_max_pooling2d_0, Parameters: 0
    Layer: nr_simple_conv2d_1, Parameters: 1168
    Layer: nr_simple_max_pooling2d_1, Parameters: 0
    Layer: nr_simple_flatten, Parameters: 0
    Layer: nr_simple_dense_0, Parameters: 1605664
    Layer: nr_simple_dense_1, Parameters: 132
    
    Total Parameters in the Model: 1607044
    


```python
##################################
# Formulating the network architecture
# for a complex CNN with no regularization
##################################
set_seed()
batch_size = 32
model_nr_complex = Sequential(name="model_nr_complex")
model_nr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="nr_complex_conv2d_0"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_0"))
model_nr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_complex_conv2d_1"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_1"))
model_nr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_complex_conv2d_2"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_2"))
model_nr_complex.add(Flatten(name="nr_complex_flatten"))
model_nr_complex.add(Dense(units=128, activation='relu', name="nr_complex_dense_0"))
model_nr_complex.add(Dense(units=num_classes, activation='softmax', name="nr_complex_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a complex CNN with no regularization
##################################
print(model_nr_complex.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_nr_complex"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ nr_complex_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_max_pooling2d_0           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_max_pooling2d_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_max_pooling2d_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">6,422,656</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ nr_complex_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">516</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,468</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,468</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a complex CNN with no regularization
##################################
model_nr_complex_layer_names = [layer.name for layer in model_nr_complex.layers]
print("Layer Names:", model_nr_complex_layer_names)

```

    Layer Names: ['nr_complex_conv2d_0', 'nr_complex_max_pooling2d_0', 'nr_complex_conv2d_1', 'nr_complex_max_pooling2d_1', 'nr_complex_conv2d_2', 'nr_complex_max_pooling2d_2', 'nr_complex_flatten', 'nr_complex_dense_0', 'nr_complex_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with no regularization
##################################
for layer in model_nr_complex.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")
        
```

    Layer: nr_complex_conv2d_0, Number of Weights: 2
    Layer: nr_complex_max_pooling2d_0, Number of Weights: 0
    Layer: nr_complex_conv2d_1, Number of Weights: 2
    Layer: nr_complex_max_pooling2d_1, Number of Weights: 0
    Layer: nr_complex_conv2d_2, Number of Weights: 2
    Layer: nr_complex_max_pooling2d_2, Number of Weights: 0
    Layer: nr_complex_flatten, Number of Weights: 0
    Layer: nr_complex_dense_0, Number of Weights: 2
    Layer: nr_complex_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of parameters
# for each model layer
# for a complex CNN with no regularization
##################################
total_parameters = 0
for layer in model_nr_complex.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: nr_complex_conv2d_0, Parameters: 160
    Layer: nr_complex_max_pooling2d_0, Parameters: 0
    Layer: nr_complex_conv2d_1, Parameters: 4640
    Layer: nr_complex_max_pooling2d_1, Parameters: 0
    Layer: nr_complex_conv2d_2, Parameters: 18496
    Layer: nr_complex_max_pooling2d_2, Parameters: 0
    Layer: nr_complex_flatten, Parameters: 0
    Layer: nr_complex_dense_0, Parameters: 6422656
    Layer: nr_complex_dense_1, Parameters: 516
    
    Total Parameters in the Model: 6446468
    

#### 1.6.2.2 CNN With Dropout Regularization <a class="anchor" id="1.6.2.2"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with dropout regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_dr_simple = Sequential(name="model_dr_simple")
model_dr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="dr_simple_conv2d_0"))
model_dr_simple.add(MaxPooling2D(pool_size=(2, 2), name="dr_simple_max_pooling2d_0"))
model_dr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_simple_conv2d_1"))
model_dr_simple.add(MaxPooling2D(pool_size=(2, 2), name="dr_simple_max_pooling2d_1"))
model_dr_simple.add(Flatten(name="dr_simple_flatten"))
model_dr_simple.add(Dense(units=32, activation='relu', name="dr_simple_dense_0"))
model_dr_simple.add(Dropout(rate=0.30, name="dr_simple_dropout"))
model_dr_simple.add(Dense(units=num_classes, activation='softmax', name="dr_simple_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a simple CNN with dropout regularization
##################################
print(model_dr_simple.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_dr_simple"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dr_simple_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">80</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_max_pooling2d_0            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">1,168</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_max_pooling2d_1            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │       <span style="color: #00af00; text-decoration-color: #00af00">1,605,664</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_simple_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">132</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,044</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,044</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a simple CNN with dropout regularization
##################################
model_dr_simple_layer_names = [layer.name for layer in model_dr_simple.layers]
print("Layer Names:", model_dr_simple_layer_names)

```

    Layer Names: ['dr_simple_conv2d_0', 'dr_simple_max_pooling2d_0', 'dr_simple_conv2d_1', 'dr_simple_max_pooling2d_1', 'dr_simple_flatten', 'dr_simple_dense_0', 'dr_simple_dropout', 'dr_simple_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with dropout regularization
##################################
for layer in model_dr_simple.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")
        
```

    Layer: dr_simple_conv2d_0, Number of Weights: 2
    Layer: dr_simple_max_pooling2d_0, Number of Weights: 0
    Layer: dr_simple_conv2d_1, Number of Weights: 2
    Layer: dr_simple_max_pooling2d_1, Number of Weights: 0
    Layer: dr_simple_flatten, Number of Weights: 0
    Layer: dr_simple_dense_0, Number of Weights: 2
    Layer: dr_simple_dropout, Number of Weights: 0
    Layer: dr_simple_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of parameters
# for each model layer
# for a simple CNN with dropout regularization
##################################
total_parameters = 0
for layer in model_dr_simple.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: dr_simple_conv2d_0, Parameters: 80
    Layer: dr_simple_max_pooling2d_0, Parameters: 0
    Layer: dr_simple_conv2d_1, Parameters: 1168
    Layer: dr_simple_max_pooling2d_1, Parameters: 0
    Layer: dr_simple_flatten, Parameters: 0
    Layer: dr_simple_dense_0, Parameters: 1605664
    Layer: dr_simple_dropout, Parameters: 0
    Layer: dr_simple_dense_1, Parameters: 132
    
    Total Parameters in the Model: 1607044
    


```python
##################################
# Formulating the network architecture
# for a complex CNN with dropout regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_dr_complex = Sequential(name="model_dr_complex")
model_dr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="dr_complex_conv2d_0"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_0"))
model_dr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_complex_conv2d_1"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_1"))
model_dr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_complex_conv2d_2"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_2"))
model_dr_complex.add(Flatten(name="dr_complex_flatten"))
model_dr_complex.add(Dense(units=128, activation='relu', name="dr_complex_dense_0"))
model_dr_complex.add(Dropout(rate=0.30, name="dr_complex_dropout"))
model_dr_complex.add(Dense(units=num_classes, activation='softmax', name="dr_complex_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a complex CNN with dropout regularization
##################################
print(model_dr_complex.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_dr_complex"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dr_complex_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_max_pooling2d_0           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_max_pooling2d_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_max_pooling2d_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">6,422,656</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dr_complex_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">516</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,468</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,468</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a complex CNN with dropout regularization
##################################
model_dr_complex_layer_names = [layer.name for layer in model_dr_complex.layers]
print("Layer Names:", model_dr_complex_layer_names)

```

    Layer Names: ['dr_complex_conv2d_0', 'dr_complex_max_pooling2d_0', 'dr_complex_conv2d_1', 'dr_complex_max_pooling2d_1', 'dr_complex_conv2d_2', 'dr_complex_max_pooling2d_2', 'dr_complex_flatten', 'dr_complex_dense_0', 'dr_complex_dropout', 'dr_complex_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with dropout regularization
##################################
for layer in model_dr_complex.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")
        
```

    Layer: dr_complex_conv2d_0, Number of Weights: 2
    Layer: dr_complex_max_pooling2d_0, Number of Weights: 0
    Layer: dr_complex_conv2d_1, Number of Weights: 2
    Layer: dr_complex_max_pooling2d_1, Number of Weights: 0
    Layer: dr_complex_conv2d_2, Number of Weights: 2
    Layer: dr_complex_max_pooling2d_2, Number of Weights: 0
    Layer: dr_complex_flatten, Number of Weights: 0
    Layer: dr_complex_dense_0, Number of Weights: 2
    Layer: dr_complex_dropout, Number of Weights: 0
    Layer: dr_complex_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of parameters
# for each model layer
# for a complex CNN with dropout regularization
##################################
total_parameters = 0
for layer in model_dr_complex.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: dr_complex_conv2d_0, Parameters: 160
    Layer: dr_complex_max_pooling2d_0, Parameters: 0
    Layer: dr_complex_conv2d_1, Parameters: 4640
    Layer: dr_complex_max_pooling2d_1, Parameters: 0
    Layer: dr_complex_conv2d_2, Parameters: 18496
    Layer: dr_complex_max_pooling2d_2, Parameters: 0
    Layer: dr_complex_flatten, Parameters: 0
    Layer: dr_complex_dense_0, Parameters: 6422656
    Layer: dr_complex_dropout, Parameters: 0
    Layer: dr_complex_dense_1, Parameters: 516
    
    Total Parameters in the Model: 6446468
    

#### 1.6.2.3 CNN With Batch Normalization Regularization <a class="anchor" id="1.6.2.3"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_bnr_simple = Sequential(name="model_bnr_simple")
model_bnr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="bnr_simple_conv2d_0"))
model_bnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="bnr_simple_max_pooling2d_0"))
model_bnr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_simple_conv2d_1"))
model_bnr_simple.add(BatchNormalization(name="bnr_simple_batch_normalization"))
model_bnr_simple.add(Activation('relu', name="bnr_simple_activation"))
model_bnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="bnr_simple_max_pooling2d_1"))
model_bnr_simple.add(Flatten(name="bnr_simple_flatten"))
model_bnr_simple.add(Dense(units=32, activation='relu', name="bnr_simple_dense_0"))
model_bnr_simple.add(Dense(units=num_classes, activation='softmax', name="bnr_simple_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a simple CNN with batch normalization regularization
##################################
print(model_bnr_simple.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_bnr_simple"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bnr_simple_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">80</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_max_pooling2d_0           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">1,168</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_batch_normalization       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │              <span style="color: #00af00; text-decoration-color: #00af00">64</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_activation (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_max_pooling2d_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │       <span style="color: #00af00; text-decoration-color: #00af00">1,605,664</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_simple_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">132</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,108</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,076</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">32</span> (128.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a simple CNN with batch normalization regularization
##################################
model_bnr_simple_layer_names = [layer.name for layer in model_bnr_simple.layers]
print("Layer Names:", model_bnr_simple_layer_names)

```

    Layer Names: ['bnr_simple_conv2d_0', 'bnr_simple_max_pooling2d_0', 'bnr_simple_conv2d_1', 'bnr_simple_batch_normalization', 'bnr_simple_activation', 'bnr_simple_max_pooling2d_1', 'bnr_simple_flatten', 'bnr_simple_dense_0', 'bnr_simple_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with batch normalization regularization
##################################
for layer in model_bnr_simple.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")

```

    Layer: bnr_simple_conv2d_0, Number of Weights: 2
    Layer: bnr_simple_max_pooling2d_0, Number of Weights: 0
    Layer: bnr_simple_conv2d_1, Number of Weights: 2
    Layer: bnr_simple_batch_normalization, Number of Weights: 4
    Layer: bnr_simple_activation, Number of Weights: 0
    Layer: bnr_simple_max_pooling2d_1, Number of Weights: 0
    Layer: bnr_simple_flatten, Number of Weights: 0
    Layer: bnr_simple_dense_0, Number of Weights: 2
    Layer: bnr_simple_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with batch normalization regularization
##################################
total_parameters = 0
for layer in model_bnr_simple.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: bnr_simple_conv2d_0, Parameters: 80
    Layer: bnr_simple_max_pooling2d_0, Parameters: 0
    Layer: bnr_simple_conv2d_1, Parameters: 1168
    Layer: bnr_simple_batch_normalization, Parameters: 64
    Layer: bnr_simple_activation, Parameters: 0
    Layer: bnr_simple_max_pooling2d_1, Parameters: 0
    Layer: bnr_simple_flatten, Parameters: 0
    Layer: bnr_simple_dense_0, Parameters: 1605664
    Layer: bnr_simple_dense_1, Parameters: 132
    
    Total Parameters in the Model: 1607108
    


```python
##################################
# Formulating the network architecture
# for a complex CNN with batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_bnr_complex = Sequential(name="model_bnr_complex")
model_bnr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="bnr_complex_conv2d_0"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_0"))
model_bnr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_complex_conv2d_1"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_1"))
model_bnr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_complex_conv2d_2"))
model_bnr_complex.add(BatchNormalization(name="bnr_complex_batch_normalization"))
model_bnr_complex.add(Activation('relu', name="bnr_complex_activation"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_2"))
model_bnr_complex.add(Flatten(name="bnr_complex_flatten"))
model_bnr_complex.add(Dense(units=128, activation='relu', name="bnr_complex_dense_0"))
model_bnr_complex.add(Dense(units=num_classes, activation='softmax', name="bnr_complex_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a complex CNN with batch normalization regularization
##################################
print(model_bnr_complex.summary())
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_bnr_complex"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bnr_complex_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_max_pooling2d_0          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_max_pooling2d_1          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_batch_normalization      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_activation (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_max_pooling2d_2          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">6,422,656</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bnr_complex_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">516</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,724</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,596</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">128</span> (512.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a complex CNN with batch normalization regularization
##################################
model_bnr_complex_layer_names = [layer.name for layer in model_bnr_complex.layers]
print("Layer Names:", model_bnr_complex_layer_names)

```

    Layer Names: ['bnr_complex_conv2d_0', 'bnr_complex_max_pooling2d_0', 'bnr_complex_conv2d_1', 'bnr_complex_max_pooling2d_1', 'bnr_complex_conv2d_2', 'bnr_complex_batch_normalization', 'bnr_complex_activation', 'bnr_complex_max_pooling2d_2', 'bnr_complex_flatten', 'bnr_complex_dense_0', 'bnr_complex_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with batch normalization regularization
##################################
for layer in model_bnr_complex.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")
        
```

    Layer: bnr_complex_conv2d_0, Number of Weights: 2
    Layer: bnr_complex_max_pooling2d_0, Number of Weights: 0
    Layer: bnr_complex_conv2d_1, Number of Weights: 2
    Layer: bnr_complex_max_pooling2d_1, Number of Weights: 0
    Layer: bnr_complex_conv2d_2, Number of Weights: 2
    Layer: bnr_complex_batch_normalization, Number of Weights: 4
    Layer: bnr_complex_activation, Number of Weights: 0
    Layer: bnr_complex_max_pooling2d_2, Number of Weights: 0
    Layer: bnr_complex_flatten, Number of Weights: 0
    Layer: bnr_complex_dense_0, Number of Weights: 2
    Layer: bnr_complex_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with batch normalization regularization
##################################
total_parameters = 0
for layer in model_bnr_complex.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: bnr_complex_conv2d_0, Parameters: 160
    Layer: bnr_complex_max_pooling2d_0, Parameters: 0
    Layer: bnr_complex_conv2d_1, Parameters: 4640
    Layer: bnr_complex_max_pooling2d_1, Parameters: 0
    Layer: bnr_complex_conv2d_2, Parameters: 18496
    Layer: bnr_complex_batch_normalization, Parameters: 256
    Layer: bnr_complex_activation, Parameters: 0
    Layer: bnr_complex_max_pooling2d_2, Parameters: 0
    Layer: bnr_complex_flatten, Parameters: 0
    Layer: bnr_complex_dense_0, Parameters: 6422656
    Layer: bnr_complex_dense_1, Parameters: 516
    
    Total Parameters in the Model: 6446724
    

#### 1.6.2.4 CNN With Dropout and Batch Normalization Regularization <a class="anchor" id="1.6.2.4"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with dropout and batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_cdrbnr_simple = Sequential(name="model_cdrbnr_simple")
model_cdrbnr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="cdrbnr_simple_conv2d_0"))
model_cdrbnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_simple_max_pooling2d_0"))
model_cdrbnr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_simple_conv2d_1"))
model_cdrbnr_simple.add(BatchNormalization(name="cdrbnr_simple_batch_normalization"))
model_cdrbnr_simple.add(Activation('relu', name="cdrbnr_simple_activation"))
model_cdrbnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_simple_max_pooling2d_1"))
model_cdrbnr_simple.add(Flatten(name="cdrbnr_simple_flatten"))
model_cdrbnr_simple.add(Dense(units=32, activation='relu', name="cdrbnr_simple_dense_0"))
model_cdrbnr_simple.add(Dropout(rate=0.30, name="cdrbnr_simple_dropout"))
model_cdrbnr_simple.add(Dense(units=num_classes, activation='softmax', name="cdrbnr_simple_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a simple CNN with dropout and batch normalization regularization
##################################
print(model_cdrbnr_simple.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_cdrbnr_simple"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ cdrbnr_simple_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │              <span style="color: #00af00; text-decoration-color: #00af00">80</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_max_pooling2d_0        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">1,168</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_batch_normalization    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │              <span style="color: #00af00; text-decoration-color: #00af00">64</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_activation             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)                         │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_max_pooling2d_1        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │       <span style="color: #00af00; text-decoration-color: #00af00">1,605,664</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_simple_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">132</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,108</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,607,076</span> (6.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">32</span> (128.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a simple CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_simple_layer_names = [layer.name for layer in model_cdrbnr_simple.layers]
print("Layer Names:", model_cdrbnr_simple_layer_names)

```

    Layer Names: ['cdrbnr_simple_conv2d_0', 'cdrbnr_simple_max_pooling2d_0', 'cdrbnr_simple_conv2d_1', 'cdrbnr_simple_batch_normalization', 'cdrbnr_simple_activation', 'cdrbnr_simple_max_pooling2d_1', 'cdrbnr_simple_flatten', 'cdrbnr_simple_dense_0', 'cdrbnr_simple_dropout', 'cdrbnr_simple_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with dropout and batch normalization regularization
##################################
for layer in model_cdrbnr_simple.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")

```

    Layer: cdrbnr_simple_conv2d_0, Number of Weights: 2
    Layer: cdrbnr_simple_max_pooling2d_0, Number of Weights: 0
    Layer: cdrbnr_simple_conv2d_1, Number of Weights: 2
    Layer: cdrbnr_simple_batch_normalization, Number of Weights: 4
    Layer: cdrbnr_simple_activation, Number of Weights: 0
    Layer: cdrbnr_simple_max_pooling2d_1, Number of Weights: 0
    Layer: cdrbnr_simple_flatten, Number of Weights: 0
    Layer: cdrbnr_simple_dense_0, Number of Weights: 2
    Layer: cdrbnr_simple_dropout, Number of Weights: 0
    Layer: cdrbnr_simple_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a simple CNN with dropout and batch normalization regularization
##################################
total_parameters = 0
for layer in model_cdrbnr_simple.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: cdrbnr_simple_conv2d_0, Parameters: 80
    Layer: cdrbnr_simple_max_pooling2d_0, Parameters: 0
    Layer: cdrbnr_simple_conv2d_1, Parameters: 1168
    Layer: cdrbnr_simple_batch_normalization, Parameters: 64
    Layer: cdrbnr_simple_activation, Parameters: 0
    Layer: cdrbnr_simple_max_pooling2d_1, Parameters: 0
    Layer: cdrbnr_simple_flatten, Parameters: 0
    Layer: cdrbnr_simple_dense_0, Parameters: 1605664
    Layer: cdrbnr_simple_dropout, Parameters: 0
    Layer: cdrbnr_simple_dense_1, Parameters: 132
    
    Total Parameters in the Model: 1607108
    


```python
##################################
# Formulating the network architecture
# for a complex CNN with dropout and batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_cdrbnr_complex = Sequential(name="model_cdrbnr_complex")
model_cdrbnr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="cdrbnr_complex_conv2d_0"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_0"))
model_cdrbnr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_complex_conv2d_1"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_1"))
model_cdrbnr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_complex_conv2d_2"))
model_cdrbnr_complex.add(BatchNormalization(name="cdrbnr_complex_batch_normalization"))
model_cdrbnr_complex.add(Activation('relu', name="cdrbnr_complex_activation"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_2"))
model_cdrbnr_complex.add(Flatten(name="cdrbnr_complex_flatten"))
model_cdrbnr_complex.add(Dense(units=128, activation='relu', name="cdrbnr_complex_dense_0"))
model_cdrbnr_complex.add(Dropout(rate=0.30, name="cdrbnr_complex_dropout"))
model_cdrbnr_complex.add(Dense(units=num_classes, activation='softmax', name="cdrbnr_complex_dense_1"))

```


```python
##################################
# Displaying the model summary
# for a complex CNN with dropout and batch normalization regularization
##################################
print(model_cdrbnr_complex.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_cdrbnr_complex"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ cdrbnr_complex_conv2d_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_max_pooling2d_0       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">113</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_max_pooling2d_1       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_batch_normalization   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_activation            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)                         │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_max_pooling2d_2       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)                       │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_dense_0 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">6,422,656</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ cdrbnr_complex_dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">516</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,724</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,446,596</span> (24.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">128</span> (512.00 B)
</pre>



    None
    


```python
##################################
# Displaying the model layers
# for a complex CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_complex_layer_names = [layer.name for layer in model_cdrbnr_complex.layers]
print("Layer Names:", model_cdrbnr_complex_layer_names)

```

    Layer Names: ['cdrbnr_complex_conv2d_0', 'cdrbnr_complex_max_pooling2d_0', 'cdrbnr_complex_conv2d_1', 'cdrbnr_complex_max_pooling2d_1', 'cdrbnr_complex_conv2d_2', 'cdrbnr_complex_batch_normalization', 'cdrbnr_complex_activation', 'cdrbnr_complex_max_pooling2d_2', 'cdrbnr_complex_flatten', 'cdrbnr_complex_dense_0', 'cdrbnr_complex_dropout', 'cdrbnr_complex_dense_1']
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with dropout and batch normalization regularization
##################################
for layer in model_cdrbnr_complex.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer: {layer.name}, Number of Weights: {len(layer.get_weights())}")

```

    Layer: cdrbnr_complex_conv2d_0, Number of Weights: 2
    Layer: cdrbnr_complex_max_pooling2d_0, Number of Weights: 0
    Layer: cdrbnr_complex_conv2d_1, Number of Weights: 2
    Layer: cdrbnr_complex_max_pooling2d_1, Number of Weights: 0
    Layer: cdrbnr_complex_conv2d_2, Number of Weights: 2
    Layer: cdrbnr_complex_batch_normalization, Number of Weights: 4
    Layer: cdrbnr_complex_activation, Number of Weights: 0
    Layer: cdrbnr_complex_max_pooling2d_2, Number of Weights: 0
    Layer: cdrbnr_complex_flatten, Number of Weights: 0
    Layer: cdrbnr_complex_dense_0, Number of Weights: 2
    Layer: cdrbnr_complex_dropout, Number of Weights: 0
    Layer: cdrbnr_complex_dense_1, Number of Weights: 2
    


```python
##################################
# Displaying the number of weights
# for each model layer
# for a complex CNN with dropout and batch normalization regularization
##################################
total_parameters = 0
for layer in model_cdrbnr_complex.layers:
    layer_parameters = layer.count_params()
    total_parameters += layer_parameters
    print(f"Layer: {layer.name}, Parameters: {layer_parameters}")
print("\nTotal Parameters in the Model:", total_parameters)

```

    Layer: cdrbnr_complex_conv2d_0, Parameters: 160
    Layer: cdrbnr_complex_max_pooling2d_0, Parameters: 0
    Layer: cdrbnr_complex_conv2d_1, Parameters: 4640
    Layer: cdrbnr_complex_max_pooling2d_1, Parameters: 0
    Layer: cdrbnr_complex_conv2d_2, Parameters: 18496
    Layer: cdrbnr_complex_batch_normalization, Parameters: 256
    Layer: cdrbnr_complex_activation, Parameters: 0
    Layer: cdrbnr_complex_max_pooling2d_2, Parameters: 0
    Layer: cdrbnr_complex_flatten, Parameters: 0
    Layer: cdrbnr_complex_dense_0, Parameters: 6422656
    Layer: cdrbnr_complex_dropout, Parameters: 0
    Layer: cdrbnr_complex_dense_1, Parameters: 516
    
    Total Parameters in the Model: 6446724
    

### 1.6.3 CNN With No Regularization Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.3"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with no regularization
##################################
set_seed()
batch_size = 32
model_nr_simple = Sequential(name="model_nr_simple")
model_nr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="nr_simple_conv2d_0"))
model_nr_simple.add(MaxPooling2D(pool_size=(2, 2), name="nr_simple_max_pooling2d_0"))
model_nr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_simple_conv2d_1"))
model_nr_simple.add(MaxPooling2D(pool_size=(2, 2), name="nr_simple_max_pooling2d_1"))
model_nr_simple.add(Flatten(name="nr_simple_flatten"))
model_nr_simple.add(Dense(units=32, activation='relu', name="nr_simple_dense_0"))
model_nr_simple.add(Dense(units=num_classes, activation='softmax', name="nr_simple_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_nr_simple.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a simple CNN with no regularization
##################################
epochs = 20
set_seed()
model_nr_simple_history = model_nr_simple.fit(train_gen, 
                                steps_per_epoch=len(train_gen)+1,
                                validation_steps=len(val_gen)+1,
                                validation_data=val_gen, 
                                epochs=epochs,
                                verbose=1,
                                callbacks=[early_stopping, reduce_lr, nr_simple_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 244ms/step - loss: 0.8697 - recall: 0.4612 - val_loss: 0.9379 - val_recall: 0.6556 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 235ms/step - loss: 0.4279 - recall: 0.8129 - val_loss: 0.8684 - val_recall: 0.6792 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 221ms/step - loss: 0.3339 - recall: 0.8637 - val_loss: 0.8071 - val_recall: 0.7239 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 223ms/step - loss: 0.3062 - recall: 0.8771 - val_loss: 0.9367 - val_recall: 0.7528 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 250ms/step - loss: 0.2505 - recall: 0.9024 - val_loss: 0.8099 - val_recall: 0.7450 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 218ms/step - loss: 0.2282 - recall: 0.9033 - val_loss: 0.7319 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 227ms/step - loss: 0.1857 - recall: 0.9301 - val_loss: 0.8285 - val_recall: 0.7783 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 224ms/step - loss: 0.1783 - recall: 0.9361 - val_loss: 0.8437 - val_recall: 0.7642 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 222ms/step - loss: 0.1366 - recall: 0.9491 - val_loss: 0.8675 - val_recall: 0.8089 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 212ms/step - loss: 0.1127 - recall: 0.9611 - val_loss: 0.7600 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 217ms/step - loss: 0.0880 - recall: 0.9663 - val_loss: 0.7769 - val_recall: 0.8177 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 216ms/step - loss: 0.1055 - recall: 0.9612 - val_loss: 0.7722 - val_recall: 0.8221 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 221ms/step - loss: 0.0787 - recall: 0.9733 - val_loss: 0.7732 - val_recall: 0.8221 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 220ms/step - loss: 0.0926 - recall: 0.9680 - val_loss: 0.7768 - val_recall: 0.8221 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 228ms/step - loss: 0.0824 - recall: 0.9691 - val_loss: 0.7803 - val_recall: 0.8212 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 224ms/step - loss: 0.0939 - recall: 0.9701 - val_loss: 0.7808 - val_recall: 0.8203 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with no regularization
# on the independent validation set
##################################
model_nr_simple_y_pred_val = model_nr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 98ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with no regularization
# on the training and validation sets
##################################
plot_training_history(model_nr_simple_history, 'Simple CNN With No Regularization : ')

```


    
![png](output_157_0.png)
    



```python
##################################
# Consolidating the predictions
# for a simple CNN with no regularization
# on the validation set
##################################
model_nr_simple_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_nr_simple_y_pred_val)))
model_nr_simple_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a simple CNN with no regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_nr_simple_y_true_val, model_nr_simple_predictions_val), columns=classes, index =classes)

##################################
# Plotting the confusion matrix
# for a simple CNN with no regularization
# on the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Simple CNN With No Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold',pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```

    WARNING:tensorflow:From D:\Github_Codes\ProjectPortfolio\Portfolio_Project_56\mdeploy_venv\Lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
    
    


    
![png](output_158_1.png)
    



```python
##################################
# Calculating the model accuracy
# for a simple CNN with no regularization
# for the entire validation set
##################################
model_nr_simple_acc_val = accuracy_score(model_nr_simple_y_true_val, model_nr_simple_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with no regularization
# for the entire validation set
##################################
model_nr_simple_results_all_val = precision_recall_fscore_support(model_nr_simple_y_true_val, model_nr_simple_predictions_val, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with no regularization
# for each category of the validation set
##################################
model_nr_simple_results_class_val = precision_recall_fscore_support(model_nr_simple_y_true_val, model_nr_simple_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with no regularization
##################################
metric_columns = ['Precision','Recall','F-Score','Support']
model_nr_simple_all_df_val = pd.concat([pd.DataFrame(list(model_nr_simple_results_class_val)).T,pd.DataFrame(list(model_nr_simple_results_all_val)).T])
model_nr_simple_all_df_val.columns = metric_columns
model_nr_simple_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Simple CNN With No Regularization : Validation Set Classification Performance')
model_nr_simple_all_df_val

```

    Simple CNN With No Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.893238</td>
      <td>0.786834</td>
      <td>0.836667</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.928571</td>
      <td>0.787879</td>
      <td>0.852459</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.624573</td>
      <td>0.685393</td>
      <td>0.653571</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.772595</td>
      <td>0.910653</td>
      <td>0.835962</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.804744</td>
      <td>0.792690</td>
      <td>0.794665</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with no regularization
##################################
model_nr_simple_model_list_val = []
model_nr_simple_measure_list_val = []
model_nr_simple_category_list_val = []
model_nr_simple_value_list_val = []
model_nr_simple_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_nr_simple_model_list_val.append('CNN_NR_Simple')
        model_nr_simple_measure_list_val.append(metric_columns[i])
        model_nr_simple_category_list_val.append(model_nr_simple_all_df_val.index[j])
        model_nr_simple_value_list_val.append(model_nr_simple_all_df_val.iloc[j,i])
        model_nr_simple_dataset_list_val.append('Validation')

model_nr_simple_all_summary_val = pd.DataFrame(zip(model_nr_simple_model_list_val,
                                                   model_nr_simple_measure_list_val,
                                                   model_nr_simple_category_list_val,
                                                   model_nr_simple_value_list_val,
                                                   model_nr_simple_dataset_list_val), 
                                               columns=['CNN.Model.Name',
                                                        'Model.Metric',
                                                        'Image.Category',
                                                        'Metric.Value',
                                                        'Data.Set'])

```


```python
##################################
# Formulating the network architecture
# for a complex CNN with no regularization
##################################
set_seed()
batch_size = 32
model_nr_complex = Sequential(name="model_nr_complex")
model_nr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="nr_complex_conv2d_0"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_0"))
model_nr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_complex_conv2d_1"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_1"))
model_nr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="nr_complex_conv2d_2"))
model_nr_complex.add(MaxPooling2D(pool_size=(2, 2), name="nr_complex_max_pooling2d_2"))
model_nr_complex.add(Flatten(name="nr_complex_flatten"))
model_nr_complex.add(Dense(units=128, activation='relu', name="nr_complex_dense_0"))
model_nr_complex.add(Dense(units=num_classes, activation='softmax', name="nr_complex_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_nr_complex.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a complex CNN with no regularization
##################################
epochs = 20
set_seed()
model_nr_complex_history = model_nr_complex.fit(train_gen, 
                                steps_per_epoch=len(train_gen)+1,
                                validation_steps=len(val_gen)+1,
                                validation_data=val_gen, 
                                epochs=epochs,
                                verbose=1,
                                callbacks=[early_stopping, reduce_lr, nr_complex_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 358ms/step - loss: 1.0913 - recall: 0.3645 - val_loss: 0.8411 - val_recall: 0.6915 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 359ms/step - loss: 0.4091 - recall: 0.8322 - val_loss: 0.8689 - val_recall: 0.6862 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 355ms/step - loss: 0.2674 - recall: 0.8948 - val_loss: 0.8096 - val_recall: 0.7327 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 368ms/step - loss: 0.2156 - recall: 0.9202 - val_loss: 0.8086 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 357ms/step - loss: 0.1748 - recall: 0.9339 - val_loss: 0.8040 - val_recall: 0.7625 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 349ms/step - loss: 0.1469 - recall: 0.9431 - val_loss: 0.7236 - val_recall: 0.7984 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 354ms/step - loss: 0.1025 - recall: 0.9621 - val_loss: 0.7801 - val_recall: 0.7993 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 358ms/step - loss: 0.0918 - recall: 0.9644 - val_loss: 0.9317 - val_recall: 0.8063 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 358ms/step - loss: 0.0861 - recall: 0.9650 - val_loss: 0.8448 - val_recall: 0.8238 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 356ms/step - loss: 0.0560 - recall: 0.9774 - val_loss: 0.8052 - val_recall: 0.8300 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 350ms/step - loss: 0.0285 - recall: 0.9933 - val_loss: 0.8621 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 357ms/step - loss: 0.0294 - recall: 0.9919 - val_loss: 0.8798 - val_recall: 0.8256 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 360ms/step - loss: 0.0235 - recall: 0.9925 - val_loss: 0.8846 - val_recall: 0.8230 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 361ms/step - loss: 0.0297 - recall: 0.9903 - val_loss: 0.8888 - val_recall: 0.8247 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 355ms/step - loss: 0.0237 - recall: 0.9951 - val_loss: 0.9018 - val_recall: 0.8230 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 355ms/step - loss: 0.0283 - recall: 0.9913 - val_loss: 0.9021 - val_recall: 0.8238 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with no regularization
# on the independent validation set
##################################
model_nr_complex_y_pred_val = model_nr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 134ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with no regularization
# on the training and validation sets
##################################
plot_training_history(model_nr_complex_history, 'Complex CNN With No Regularization : ')

```


    
![png](output_164_0.png)
    



```python
##################################
# Consolidating the predictions
# for a complex CNN with no regularization
# on the validation set
##################################
model_nr_complex_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_nr_complex_y_pred_val)))
model_nr_complex_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a complex CNN with no regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_nr_complex_y_true_val, model_nr_complex_predictions_val), columns=classes, index =classes)

##################################
# Plotting the confusion matrix
# for a complex CNN with no regularization
# on the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Complex CNN With No Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold',pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_165_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a complex CNN with no regularization
# for the entire validation set
##################################
model_nr_complex_acc_val = accuracy_score(model_nr_complex_y_true_val, model_nr_complex_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with no regularization
# for the entire validation set
##################################
model_nr_complex_results_all_val = precision_recall_fscore_support(model_nr_complex_y_true_val, model_nr_complex_predictions_val, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with no regularization
# for each category of the validation set
##################################
model_nr_complex_results_class_val = precision_recall_fscore_support(model_nr_complex_y_true_val, model_nr_complex_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with no regularization
##################################
metric_columns = ['Precision','Recall','F-Score','Support']
model_nr_complex_all_df_val = pd.concat([pd.DataFrame(list(model_nr_complex_results_class_val)).T,pd.DataFrame(list(model_nr_complex_results_all_val)).T])
model_nr_complex_all_df_val.columns = metric_columns
model_nr_complex_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Complex CNN With No Regularization : Validation Set Classification Performance')
model_nr_complex_all_df_val

```

    Complex CNN With No Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.863057</td>
      <td>0.849530</td>
      <td>0.856240</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.871486</td>
      <td>0.821970</td>
      <td>0.846004</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.655602</td>
      <td>0.591760</td>
      <td>0.622047</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.795252</td>
      <td>0.920962</td>
      <td>0.853503</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.796349</td>
      <td>0.796055</td>
      <td>0.794449</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with no regularization
##################################
model_nr_complex_model_list_val = []
model_nr_complex_measure_list_val = []
model_nr_complex_category_list_val = []
model_nr_complex_value_list_val = []
model_nr_complex_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_nr_complex_model_list_val.append('CNN_NR_Complex')
        model_nr_complex_measure_list_val.append(metric_columns[i])
        model_nr_complex_category_list_val.append(model_nr_complex_all_df_val.index[j])
        model_nr_complex_value_list_val.append(model_nr_complex_all_df_val.iloc[j,i])
        model_nr_complex_dataset_list_val.append('Validation')

model_nr_complex_all_summary_val = pd.DataFrame(zip(model_nr_complex_model_list_val,
                                                    model_nr_complex_measure_list_val,
                                                    model_nr_complex_category_list_val,
                                                    model_nr_complex_value_list_val,
                                                    model_nr_complex_dataset_list_val), 
                                                columns=['CNN.Model.Name',
                                                         'Model.Metric',
                                                         'Image.Category',
                                                         'Metric.Value',
                                                         'Data.Set'])

```

### 1.6.4 CNN With Dropout Regularization Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.4"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with dropout regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_dr_simple = Sequential(name="model_dr_simple")
model_dr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="dr_simple_conv2d_0"))
model_dr_simple.add(MaxPooling2D(pool_size=(2, 2), name="dr_simple_max_pooling2d_0"))
model_dr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_simple_conv2d_1"))
model_dr_simple.add(MaxPooling2D(pool_size=(2, 2), name="dr_simple_max_pooling2d_1"))
model_dr_simple.add(Flatten(name="dr_simple_flatten"))
model_dr_simple.add(Dense(units=32, activation='relu', name="dr_simple_dense_0"))
model_dr_simple.add(Dropout(rate=0.30, name="dr_simple_dropout"))
model_dr_simple.add(Dense(units=num_classes, activation='softmax', name="dr_simple_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_dr_simple.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a simple CNN with dropout regularization
##################################
epochs = 20
set_seed()
model_dr_simple_history = model_dr_simple.fit(train_gen, 
                                steps_per_epoch=len(train_gen)+1,
                                validation_steps=len(val_gen)+1,
                                validation_data=val_gen, 
                                epochs=epochs,
                                verbose=1,
                                callbacks=[early_stopping, reduce_lr, dr_simple_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 238ms/step - loss: 1.3558 - recall: 0.1436 - val_loss: 1.0029 - val_recall: 0.4259 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 218ms/step - loss: 0.7573 - recall: 0.5541 - val_loss: 0.8809 - val_recall: 0.5995 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 220ms/step - loss: 0.6801 - recall: 0.5991 - val_loss: 0.8098 - val_recall: 0.6784 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 218ms/step - loss: 0.5949 - recall: 0.6555 - val_loss: 0.9510 - val_recall: 0.6319 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 219ms/step - loss: 0.5358 - recall: 0.6888 - val_loss: 0.8406 - val_recall: 0.6687 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 221ms/step - loss: 0.5175 - recall: 0.7039 - val_loss: 0.7385 - val_recall: 0.6950 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 223ms/step - loss: 0.5096 - recall: 0.7264 - val_loss: 0.8432 - val_recall: 0.7108 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m33s[0m 226ms/step - loss: 0.5263 - recall: 0.7275 - val_loss: 0.7060 - val_recall: 0.7432 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 224ms/step - loss: 0.4338 - recall: 0.7747 - val_loss: 0.8316 - val_recall: 0.7546 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 216ms/step - loss: 0.4617 - recall: 0.7647 - val_loss: 0.8108 - val_recall: 0.7432 - learning_rate: 0.0010
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 219ms/step - loss: 0.4197 - recall: 0.7834 - val_loss: 0.8501 - val_recall: 0.7406 - learning_rate: 0.0010
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 219ms/step - loss: 0.4121 - recall: 0.7925 - val_loss: 0.7721 - val_recall: 0.7634 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 221ms/step - loss: 0.3817 - recall: 0.8064 - val_loss: 0.7482 - val_recall: 0.7713 - learning_rate: 1.0000e-04
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 226ms/step - loss: 0.3763 - recall: 0.8102 - val_loss: 0.7683 - val_recall: 0.7634 - learning_rate: 1.0000e-04
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 222ms/step - loss: 0.3781 - recall: 0.7994 - val_loss: 0.7877 - val_recall: 0.7642 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 223ms/step - loss: 0.3701 - recall: 0.8088 - val_loss: 0.7936 - val_recall: 0.7642 - learning_rate: 1.0000e-05
    Epoch 17/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 221ms/step - loss: 0.3933 - recall: 0.8056 - val_loss: 0.7841 - val_recall: 0.7660 - learning_rate: 1.0000e-05
    Epoch 18/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m31s[0m 216ms/step - loss: 0.3852 - recall: 0.7920 - val_loss: 0.7832 - val_recall: 0.7660 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with dropout regularization
# on the independent validation set
##################################
model_dr_simple_y_pred_val = model_dr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 91ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with dropout regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_simple_history, 'Simple CNN With Dropout Regularization : ')

```


    
![png](output_172_0.png)
    



```python
##################################
# Consolidating the predictions
# for a simple CNN with dropout regularization
# on the validation set
##################################
model_dr_simple_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_dr_simple_y_pred_val)))
model_dr_simple_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a simple CNN with dropout regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_dr_simple_y_true_val, model_dr_simple_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Simple CNN With Dropout Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_173_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a simple CNN with dropout regularization
# for the entire validation set
##################################
model_dr_simple_acc_val = accuracy_score(model_dr_simple_y_true_val, model_dr_simple_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout regularization
# for the entire validation set
##################################
model_dr_simple_results_all_val = precision_recall_fscore_support(model_dr_simple_y_true_val, model_dr_simple_predictions_val, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout regularization
# for each category of the validation set
##################################
model_dr_simple_results_class_val = precision_recall_fscore_support(model_dr_simple_y_true_val, model_dr_simple_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with dropout regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_dr_simple_all_df_val = pd.concat([pd.DataFrame(list(model_dr_simple_results_class_val)).T,pd.DataFrame(list(model_dr_simple_results_all_val)).T])
model_dr_simple_all_df_val.columns = metric_columns
model_dr_simple_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Simple CNN With Dropout Regularization : Validation Set Classification Performance')
model_dr_simple_all_df_val

```

    Simple CNN With Dropout Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.852459</td>
      <td>0.815047</td>
      <td>0.833333</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.902778</td>
      <td>0.738636</td>
      <td>0.812500</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.554054</td>
      <td>0.614232</td>
      <td>0.582593</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.774691</td>
      <td>0.862543</td>
      <td>0.816260</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.770996</td>
      <td>0.757615</td>
      <td>0.761172</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with dropout regularization
##################################
model_dr_simple_model_list_val = []
model_dr_simple_measure_list_val = []
model_dr_simple_category_list_val = []
model_dr_simple_value_list_val = []
model_dr_simple_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_dr_simple_model_list_val.append('CNN_DR_Simple')
        model_dr_simple_measure_list_val.append(metric_columns[i])
        model_dr_simple_category_list_val.append(model_dr_simple_all_df_val.index[j])
        model_dr_simple_value_list_val.append(model_dr_simple_all_df_val.iloc[j,i])
        model_dr_simple_dataset_list_val.append('Validation')

model_dr_simple_all_summary_val = pd.DataFrame(zip(model_dr_simple_model_list_val,
                                                   model_dr_simple_measure_list_val,
                                                   model_dr_simple_category_list_val,
                                                   model_dr_simple_value_list_val,
                                                   model_dr_simple_dataset_list_val), 
                                               columns=['CNN.Model.Name',
                                                        'Model.Metric',
                                                        'Image.Category',
                                                        'Metric.Value',
                                                        'Data.Set'])

```


```python
##################################
# Formulating the network architecture
# for a complex CNN with dropout regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_dr_complex = Sequential(name="model_dr_complex")
model_dr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="dr_complex_conv2d_0"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_0"))
model_dr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_complex_conv2d_1"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_1"))
model_dr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="dr_complex_conv2d_2"))
model_dr_complex.add(MaxPooling2D(pool_size=(2, 2), name="dr_complex_max_pooling2d_2"))
model_dr_complex.add(Flatten(name="dr_complex_flatten"))
model_dr_complex.add(Dense(units=128, activation='relu', name="dr_complex_dense_0"))
model_dr_complex.add(Dropout(rate=0.30, name="dr_complex_dropout"))
model_dr_complex.add(Dense(units=num_classes, activation='softmax', name="dr_complex_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_dr_complex.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a complex CNN with dropout regularization
##################################
epochs = 20
set_seed()
model_dr_complex_history = model_dr_complex.fit(train_gen, 
                                steps_per_epoch=len(train_gen)+1,
                                validation_steps=len(val_gen)+1,
                                validation_data=val_gen, 
                                epochs=epochs,
                                verbose=1,
                                callbacks=[early_stopping, reduce_lr, dr_complex_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 363ms/step - loss: 1.0131 - recall: 0.3707 - val_loss: 0.8088 - val_recall: 0.6994 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 357ms/step - loss: 0.4345 - recall: 0.8110 - val_loss: 0.7967 - val_recall: 0.6968 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 357ms/step - loss: 0.2910 - recall: 0.8898 - val_loss: 0.7494 - val_recall: 0.7458 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 350ms/step - loss: 0.2426 - recall: 0.9008 - val_loss: 0.7891 - val_recall: 0.7511 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 356ms/step - loss: 0.1822 - recall: 0.9304 - val_loss: 0.6271 - val_recall: 0.7844 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 351ms/step - loss: 0.1632 - recall: 0.9328 - val_loss: 0.7265 - val_recall: 0.7774 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 353ms/step - loss: 0.1317 - recall: 0.9478 - val_loss: 0.8423 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 352ms/step - loss: 0.1286 - recall: 0.9583 - val_loss: 0.8516 - val_recall: 0.8107 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 352ms/step - loss: 0.0860 - recall: 0.9707 - val_loss: 0.7973 - val_recall: 0.8124 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 351ms/step - loss: 0.0758 - recall: 0.9745 - val_loss: 0.8234 - val_recall: 0.8081 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 349ms/step - loss: 0.0523 - recall: 0.9825 - val_loss: 0.8551 - val_recall: 0.8098 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 358ms/step - loss: 0.0571 - recall: 0.9813 - val_loss: 0.8562 - val_recall: 0.8054 - learning_rate: 1.0000e-05
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 355ms/step - loss: 0.0540 - recall: 0.9823 - val_loss: 0.8620 - val_recall: 0.8089 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 355ms/step - loss: 0.0564 - recall: 0.9793 - val_loss: 0.8652 - val_recall: 0.8098 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m51s[0m 352ms/step - loss: 0.0538 - recall: 0.9812 - val_loss: 0.8655 - val_recall: 0.8098 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with dropout regularization
# on the independent validation set
##################################
model_dr_complex_y_pred_val = model_dr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 135ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with dropout regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_complex_history, 'Complex CNN With Dropout Regularization : ')

```


    
![png](output_179_0.png)
    



```python
##################################
# Consolidating the predictions
# for a complex CNN with dropout regularization
# on the validation set
##################################
model_dr_complex_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_dr_complex_y_pred_val)))
model_dr_complex_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a complex CNN with dropout regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_dr_complex_y_true_val, model_dr_complex_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Complex CNN With Dropout Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_180_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a complex CNN with dropout regularization
# for the entire validation set
##################################
model_dr_complex_acc_val = accuracy_score(model_dr_complex_y_true_val, model_dr_complex_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout regularization
# for the entire validation set
##################################
model_dr_complex_results_all_val = precision_recall_fscore_support(model_dr_complex_y_true_val, model_dr_complex_predictions_val, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout regularization
# for each category of the validation set
##################################
model_dr_complex_results_class_val = precision_recall_fscore_support(model_dr_complex_y_true_val, model_dr_complex_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with dropout regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_dr_complex_all_df_val = pd.concat([pd.DataFrame(list(model_dr_complex_results_class_val)).T,pd.DataFrame(list(model_dr_complex_results_all_val)).T])
model_dr_complex_all_df_val.columns = metric_columns
model_dr_complex_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Complex CNN With Dropout Regularization : Validation Set Classification Performance')
model_dr_complex_all_df_val

```

    Complex CNN With Dropout Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.835913</td>
      <td>0.846395</td>
      <td>0.841121</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.858238</td>
      <td>0.848485</td>
      <td>0.853333</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.641975</td>
      <td>0.584270</td>
      <td>0.611765</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.815287</td>
      <td>0.879725</td>
      <td>0.846281</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.787853</td>
      <td>0.789719</td>
      <td>0.788125</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with dropout regularization
##################################
model_dr_complex_model_list_val = []
model_dr_complex_measure_list_val = []
model_dr_complex_category_list_val = []
model_dr_complex_value_list_val = []
model_dr_complex_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_dr_complex_model_list_val.append('CNN_DR_Complex')
        model_dr_complex_measure_list_val.append(metric_columns[i])
        model_dr_complex_category_list_val.append(model_dr_complex_all_df_val.index[j])
        model_dr_complex_value_list_val.append(model_dr_complex_all_df_val.iloc[j,i])
        model_dr_complex_dataset_list_val.append('Validation')

model_dr_complex_all_summary_val = pd.DataFrame(zip(model_dr_complex_model_list_val,
                                                    model_dr_complex_measure_list_val,
                                                    model_dr_complex_category_list_val,
                                                    model_dr_complex_value_list_val,
                                                    model_dr_complex_dataset_list_val), 
                                                columns=['CNN.Model.Name',
                                                         'Model.Metric',
                                                         'Image.Category',
                                                         'Metric.Value',
                                                         'Data.Set'])

```

### 1.6.5 CNN With Batch Normalization Regularization Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.5"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_bnr_simple = Sequential(name="model_bnr_simple")
model_bnr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="bnr_simple_conv2d_0"))
model_bnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="bnr_simple_max_pooling2d_0"))
model_bnr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_simple_conv2d_1"))
model_bnr_simple.add(BatchNormalization(name="bnr_simple_batch_normalization"))
model_bnr_simple.add(Activation('relu', name="bnr_simple_activation"))
model_bnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="bnr_simple_max_pooling2d_1"))
model_bnr_simple.add(Flatten(name="bnr_simple_flatten"))
model_bnr_simple.add(Dense(units=32, activation='relu', name="bnr_simple_dense_0"))
model_bnr_simple.add(Dense(units=num_classes, activation='softmax', name="bnr_simple_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_bnr_simple.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a simple CNN with batch normalization regularization
##################################
epochs = 20
set_seed()
model_bnr_simple_history = model_bnr_simple.fit(train_gen, 
                                  steps_per_epoch=len(train_gen)+1,
                                  validation_steps=len(val_gen)+1,
                                  validation_data=val_gen, 
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[early_stopping, reduce_lr, bnr_simple_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 263ms/step - loss: 1.7668 - recall: 0.5558 - val_loss: 1.0888 - val_recall: 0.0473 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 262ms/step - loss: 0.3585 - recall: 0.8676 - val_loss: 0.8608 - val_recall: 0.3716 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 250ms/step - loss: 0.2334 - recall: 0.9148 - val_loss: 0.7054 - val_recall: 0.6591 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 255ms/step - loss: 0.2119 - recall: 0.9237 - val_loss: 0.5743 - val_recall: 0.8089 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.2065 - recall: 0.9280 - val_loss: 0.6802 - val_recall: 0.8072 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.1448 - recall: 0.9461 - val_loss: 0.8415 - val_recall: 0.8387 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 257ms/step - loss: 0.1309 - recall: 0.9561 - val_loss: 1.1974 - val_recall: 0.8107 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.0820 - recall: 0.9706 - val_loss: 0.9800 - val_recall: 0.8282 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 256ms/step - loss: 0.0649 - recall: 0.9801 - val_loss: 1.0222 - val_recall: 0.8309 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.0683 - recall: 0.9782 - val_loss: 1.0025 - val_recall: 0.8247 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 261ms/step - loss: 0.0509 - recall: 0.9825 - val_loss: 0.9991 - val_recall: 0.8309 - learning_rate: 1.0000e-05
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.0648 - recall: 0.9745 - val_loss: 0.9882 - val_recall: 0.8309 - learning_rate: 1.0000e-05
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 255ms/step - loss: 0.0472 - recall: 0.9859 - val_loss: 0.9759 - val_recall: 0.8300 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 262ms/step - loss: 0.0499 - recall: 0.9851 - val_loss: 0.9774 - val_recall: 0.8309 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with batch normalization regularization
# on the independent validation set
##################################
model_bnr_simple_y_pred_val = model_bnr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 98ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_bnr_simple_history, 'Simple CNN With Batch Normalization Regularization : ')

```


    
![png](output_187_0.png)
    



```python
##################################
# Consolidating the predictions
# for a simple CNN with batch normalization regularization
# on the validation set
##################################
model_bnr_simple_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_bnr_simple_y_pred_val)))
model_bnr_simple_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a simple CNN with batch normalization regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_bnr_simple_y_true_val, model_bnr_simple_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Simple CNN With Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_188_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a simple CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_simple_acc_val = accuracy_score(model_bnr_simple_y_true_val, model_bnr_simple_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_simple_results_all_val = precision_recall_fscore_support(model_bnr_simple_y_true_val, model_bnr_simple_predictions_val, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for each category of the validation set
##################################
model_bnr_simple_results_class_val = precision_recall_fscore_support(model_bnr_simple_y_true_val, model_bnr_simple_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_bnr_simple_all_df_val = pd.concat([pd.DataFrame(list(model_bnr_simple_results_class_val)).T,pd.DataFrame(list(model_bnr_simple_results_all_val)).T])
model_bnr_simple_all_df_val.columns = metric_columns
model_bnr_simple_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Simple CNN With Batch Normalization Regularization : Validation Set Classification Performance')
model_bnr_simple_all_df_val

```

    Simple CNN With Batch Normalization Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.832317</td>
      <td>0.855799</td>
      <td>0.843895</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.962025</td>
      <td>0.863636</td>
      <td>0.910180</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.690647</td>
      <td>0.719101</td>
      <td>0.704587</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.845638</td>
      <td>0.865979</td>
      <td>0.855688</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.832657</td>
      <td>0.826129</td>
      <td>0.828587</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with batch normalization regularization
##################################
model_bnr_simple_model_list_val = []
model_bnr_simple_measure_list_val = []
model_bnr_simple_category_list_val = []
model_bnr_simple_value_list_val = []
model_bnr_simple_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_bnr_simple_model_list_val.append('CNN_BNR_Simple')
        model_bnr_simple_measure_list_val.append(metric_columns[i])
        model_bnr_simple_category_list_val.append(model_bnr_simple_all_df_val.index[j])
        model_bnr_simple_value_list_val.append(model_bnr_simple_all_df_val.iloc[j,i])
        model_bnr_simple_dataset_list_val.append('Validation')

model_bnr_simple_all_summary_val = pd.DataFrame(zip(model_bnr_simple_model_list_val,
                                                    model_bnr_simple_measure_list_val,
                                                    model_bnr_simple_category_list_val,
                                                    model_bnr_simple_value_list_val,
                                                    model_bnr_simple_dataset_list_val), 
                                                columns=['CNN.Model.Name',
                                                         'Model.Metric',
                                                         'Image.Category',
                                                         'Metric.Value',
                                                         'Data.Set'])

```


```python
##################################
# Formulating the network architecture
# for a complex CNN with batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_bnr_complex = Sequential(name="model_bnr_complex")
model_bnr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="bnr_complex_conv2d_0"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_0"))
model_bnr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_complex_conv2d_1"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_1"))
model_bnr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="bnr_complex_conv2d_2"))
model_bnr_complex.add(BatchNormalization(name="bnr_complex_batch_normalization"))
model_bnr_complex.add(Activation('relu', name="bnr_complex_activation"))
model_bnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="bnr_complex_max_pooling2d_2"))
model_bnr_complex.add(Flatten(name="bnr_complex_flatten"))
model_bnr_complex.add(Dense(units=128, activation='relu', name="bnr_complex_dense_0"))
model_bnr_complex.add(Dense(units=num_classes, activation='softmax', name="bnr_complex_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_bnr_complex.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])
```


```python
##################################
# Fitting the model
# for a complex CNN with batch normalization regularization
##################################
epochs = 20
set_seed()
model_bnr_complex_history = model_bnr_complex.fit(train_gen, 
                                  steps_per_epoch=len(train_gen)+1,
                                  validation_steps=len(val_gen)+1,
                                  validation_data=val_gen, 
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[early_stopping, reduce_lr, bnr_complex_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 387ms/step - loss: 2.4198 - recall: 0.4782 - val_loss: 1.1481 - val_recall: 0.0096 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 381ms/step - loss: 0.3966 - recall: 0.8304 - val_loss: 0.9454 - val_recall: 0.1613 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 380ms/step - loss: 0.2384 - recall: 0.9055 - val_loss: 0.7357 - val_recall: 0.5819 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 386ms/step - loss: 0.2179 - recall: 0.9136 - val_loss: 0.6788 - val_recall: 0.7809 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 380ms/step - loss: 0.1693 - recall: 0.9332 - val_loss: 0.8541 - val_recall: 0.7064 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 376ms/step - loss: 0.1205 - recall: 0.9529 - val_loss: 0.8922 - val_recall: 0.7774 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 382ms/step - loss: 0.1140 - recall: 0.9631 - val_loss: 1.1084 - val_recall: 0.7695 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.0670 - recall: 0.9783 - val_loss: 0.8778 - val_recall: 0.8151 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 384ms/step - loss: 0.0429 - recall: 0.9854 - val_loss: 0.8952 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.0439 - recall: 0.9852 - val_loss: 0.8729 - val_recall: 0.8335 - learning_rate: 1.0000e-04
    


```python
##################################
# Evaluating the model
# for a complex CNN with batch normalization regularization
# on the independent validation set
##################################
model_bnr_complex_y_pred_val = model_bnr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 138ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_bnr_complex_history, 'Complex CNN With Batch Normalization Regularization : ')

```


    
![png](output_194_0.png)
    



```python
##################################
# Consolidating the predictions
# for a complex CNN with batch normalization regularization
# on the validation set
##################################
model_bnr_complex_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_bnr_complex_y_pred_val)))
model_bnr_complex_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a complex CNN with batch normalization regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_bnr_complex_y_true_val, model_bnr_complex_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Complex CNN With Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_195_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a complex CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_complex_acc_val = accuracy_score(model_bnr_complex_y_true_val, model_bnr_complex_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_complex_results_all_val = precision_recall_fscore_support(model_bnr_complex_y_true_val, model_bnr_complex_predictions_val, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with batch normalization regularization
# for each category of the validation set
##################################
model_bnr_complex_results_class_val = precision_recall_fscore_support(model_bnr_complex_y_true_val, model_bnr_complex_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_bnr_complex_all_df_val = pd.concat([pd.DataFrame(list(model_bnr_complex_results_class_val)).T,pd.DataFrame(list(model_bnr_complex_results_all_val)).T])
model_bnr_complex_all_df_val.columns = metric_columns
model_bnr_complex_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Complex CNN With Batch Normalization Regularization : Validation Set Classification Performance')
model_bnr_complex_all_df_val

```

    Complex CNN With Batch Normalization Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.833333</td>
      <td>0.611285</td>
      <td>0.705244</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.623684</td>
      <td>0.897727</td>
      <td>0.736025</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.430070</td>
      <td>0.460674</td>
      <td>0.444846</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.780083</td>
      <td>0.646048</td>
      <td>0.706767</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.666793</td>
      <td>0.653934</td>
      <td>0.648221</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with batch normalization regularization
##################################
model_bnr_complex_model_list_val = []
model_bnr_complex_measure_list_val = []
model_bnr_complex_category_list_val = []
model_bnr_complex_value_list_val = []
model_bnr_complex_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_bnr_complex_model_list_val.append('CNN_BNR_Complex')
        model_bnr_complex_measure_list_val.append(metric_columns[i])
        model_bnr_complex_category_list_val.append(model_bnr_complex_all_df_val.index[j])
        model_bnr_complex_value_list_val.append(model_bnr_complex_all_df_val.iloc[j,i])
        model_bnr_complex_dataset_list_val.append('Validation')

model_bnr_complex_all_summary_val = pd.DataFrame(zip(model_bnr_complex_model_list_val,
                                                     model_bnr_complex_measure_list_val,
                                                     model_bnr_complex_category_list_val,
                                                     model_bnr_complex_value_list_val,
                                                     model_bnr_complex_dataset_list_val), 
                                                 columns=['CNN.Model.Name',
                                                          'Model.Metric',
                                                          'Image.Category',
                                                          'Metric.Value',
                                                          'Data.Set'])

```

### 1.6.6 CNN With Dropout and Batch Normalization Regularization Model Fitting | Hyperparameter Tuning | Validation <a class="anchor" id="1.6.6"></a>


```python
##################################
# Formulating the network architecture
# for a simple CNN with dropout and batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_cdrbnr_simple = Sequential(name="model_cdrbnr_simple")
model_cdrbnr_simple.add(Conv2D(filters=8, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="cdrbnr_simple_conv2d_0"))
model_cdrbnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_simple_max_pooling2d_0"))
model_cdrbnr_simple.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_simple_conv2d_1"))
model_cdrbnr_simple.add(BatchNormalization(name="cdrbnr_simple_batch_normalization"))
model_cdrbnr_simple.add(Activation('relu', name="cdrbnr_simple_activation"))
model_cdrbnr_simple.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_simple_max_pooling2d_1"))
model_cdrbnr_simple.add(Flatten(name="cdrbnr_simple_flatten"))
model_cdrbnr_simple.add(Dense(units=32, activation='relu', name="cdrbnr_simple_dense_0"))
model_cdrbnr_simple.add(Dropout(rate=0.30, name="cdrbnr_simple_dropout"))
model_cdrbnr_simple.add(Dense(units=num_classes, activation='softmax', name="cdrbnr_simple_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_cdrbnr_simple.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a simple CNN with dropout and batch normalization regularization
##################################
epochs = 20
set_seed()
model_cdrbnr_simple_history = model_cdrbnr_simple.fit(train_gen, 
                                  steps_per_epoch=len(train_gen)+1,
                                  validation_steps=len(val_gen)+1,
                                  validation_data=val_gen, 
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[early_stopping, reduce_lr, cdrbnr_simple_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 257ms/step - loss: 1.6579 - recall: 0.1515 - val_loss: 1.3345 - val_recall: 0.0018 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 270ms/step - loss: 1.0206 - recall: 0.3417 - val_loss: 1.1807 - val_recall: 0.0649 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 260ms/step - loss: 0.9324 - recall: 0.3955 - val_loss: 1.0523 - val_recall: 0.2366 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 268ms/step - loss: 0.7758 - recall: 0.4966 - val_loss: 0.9607 - val_recall: 0.4137 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.7319 - recall: 0.5117 - val_loss: 1.0513 - val_recall: 0.4496 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 255ms/step - loss: 0.6944 - recall: 0.5397 - val_loss: 1.0002 - val_recall: 0.5127 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.6810 - recall: 0.5275 - val_loss: 1.1606 - val_recall: 0.6056 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.6298 - recall: 0.5520 - val_loss: 0.9720 - val_recall: 0.5951 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.5942 - recall: 0.5613 - val_loss: 0.9829 - val_recall: 0.5960 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 256ms/step - loss: 0.6268 - recall: 0.5480 - val_loss: 1.0679 - val_recall: 0.5942 - learning_rate: 1.0000e-04
    


```python
##################################
# Evaluating the model
# for a simple CNN with dropout and batch normalization regularization
# on the independent validation set
##################################
model_cdrbnr_simple_y_pred_val = model_cdrbnr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 101ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with dropout and batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_cdrbnr_simple_history, 'Simple CNN With Dropout and Batch Normalization Regularization : ')

```


    
![png](output_202_0.png)
    



```python
##################################
# Consolidating the predictions
# for a simple CNN with dropout and batch normalization regularization
# on the validation set
##################################
model_cdrbnr_simple_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_cdrbnr_simple_y_pred_val)))
model_cdrbnr_simple_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a simple CNN with dropout and batch normalization regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_cdrbnr_simple_y_true_val, model_cdrbnr_simple_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout and batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Simple CNN With Dropout and Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_203_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a simple CNN with dropout and batch normalization regularization
# for the entire validation set
##################################
model_cdrbnr_simple_acc_val = accuracy_score(model_cdrbnr_simple_y_true_val, model_cdrbnr_simple_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout and batch normalization regularization
# for the entire validation set
##################################
model_cdrbnr_simple_results_all_val = precision_recall_fscore_support(model_cdrbnr_simple_y_true_val, model_cdrbnr_simple_predictions_val, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with dropout and batch normalization regularization
# for each category of the validation set
##################################
model_cdrbnr_simple_results_class_val = precision_recall_fscore_support(model_cdrbnr_simple_y_true_val, model_cdrbnr_simple_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with dropout and batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_cdrbnr_simple_all_df_val = pd.concat([pd.DataFrame(list(model_cdrbnr_simple_results_class_val)).T,pd.DataFrame(list(model_cdrbnr_simple_results_all_val)).T])
model_cdrbnr_simple_all_df_val.columns = metric_columns
model_cdrbnr_simple_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Simple CNN With Dropout and Batch Normalization Regularization : Validation Set Classification Performance')
model_cdrbnr_simple_all_df_val

```

    Simple CNN With Dropout and Batch Normalization Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.722656</td>
      <td>0.579937</td>
      <td>0.643478</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.875000</td>
      <td>0.026515</td>
      <td>0.051471</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.313199</td>
      <td>0.524345</td>
      <td>0.392157</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.490698</td>
      <td>0.725086</td>
      <td>0.585298</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.600388</td>
      <td>0.463971</td>
      <td>0.418101</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_simple_model_list_val = []
model_cdrbnr_simple_measure_list_val = []
model_cdrbnr_simple_category_list_val = []
model_cdrbnr_simple_value_list_val = []
model_cdrbnr_simple_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_cdrbnr_simple_model_list_val.append('CNN_CDRBNR_Simple')
        model_cdrbnr_simple_measure_list_val.append(metric_columns[i])
        model_cdrbnr_simple_category_list_val.append(model_cdrbnr_simple_all_df_val.index[j])
        model_cdrbnr_simple_value_list_val.append(model_cdrbnr_simple_all_df_val.iloc[j,i])
        model_cdrbnr_simple_dataset_list_val.append('Validation')

model_cdrbnr_simple_all_summary_val = pd.DataFrame(zip(model_cdrbnr_simple_model_list_val,
                                                       model_cdrbnr_simple_measure_list_val,
                                                       model_cdrbnr_simple_category_list_val,
                                                       model_cdrbnr_simple_value_list_val,
                                                       model_cdrbnr_simple_dataset_list_val), 
                                                   columns=['CNN.Model.Name',
                                                            'Model.Metric',
                                                            'Image.Category',
                                                            'Metric.Value',
                                                            'Data.Set'])

```


```python
##################################
# Formulating the network architecture
# for a complex CNN with dropout and batch normalization regularization
##################################
set_seed()
batch_size = 32
input_shape = (227, 227, 1)
model_cdrbnr_complex = Sequential(name="model_cdrbnr_complex")
model_cdrbnr_complex.add(Conv2D(filters=16, kernel_size=(3, 3), padding = 'Same', activation='relu', input_shape=(227, 227, 1), name="cdrbnr_complex_conv2d_0"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_0"))
model_cdrbnr_complex.add(Conv2D(filters=32, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_complex_conv2d_1"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_1"))
model_cdrbnr_complex.add(Conv2D(filters=64, kernel_size=(3, 3), padding = 'Same', activation='relu', name="cdrbnr_complex_conv2d_2"))
model_cdrbnr_complex.add(BatchNormalization(name="cdrbnr_complex_batch_normalization"))
model_cdrbnr_complex.add(Activation('relu', name="cdrbnr_complex_activation"))
model_cdrbnr_complex.add(MaxPooling2D(pool_size=(2, 2), name="cdrbnr_complex_max_pooling2d_2"))
model_cdrbnr_complex.add(Flatten(name="cdrbnr_complex_flatten"))
model_cdrbnr_complex.add(Dense(units=128, activation='relu', name="cdrbnr_complex_dense_0"))
model_cdrbnr_complex.add(Dropout(rate=0.30, name="cdrbnr_complex_dropout"))
model_cdrbnr_complex.add(Dense(units=num_classes, activation='softmax', name="cdrbnr_complex_dense_1"))

##################################
# Compiling the network layers
##################################
optimizer = Adam(learning_rate=0.001)
model_cdrbnr_complex.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Fitting the model
# for a complex CNN with dropout and batch normalization regularization
##################################
epochs = 20
set_seed()
model_cdrbnr_complex_history = model_cdrbnr_complex.fit(train_gen, 
                                  steps_per_epoch=len(train_gen)+1,
                                  validation_steps=len(val_gen)+1,
                                  validation_data=val_gen, 
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[early_stopping, reduce_lr, cdrbnr_complex_model_checkpoint])

```

    Epoch 1/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 386ms/step - loss: 1.7995 - recall: 0.5219 - val_loss: 1.1321 - val_recall: 0.0342 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 381ms/step - loss: 0.3938 - recall: 0.8333 - val_loss: 0.9887 - val_recall: 0.0649 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 381ms/step - loss: 0.2484 - recall: 0.8988 - val_loss: 0.6290 - val_recall: 0.6713 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 382ms/step - loss: 0.2268 - recall: 0.9093 - val_loss: 0.6252 - val_recall: 0.7555 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 377ms/step - loss: 0.1590 - recall: 0.9359 - val_loss: 0.8430 - val_recall: 0.7046 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m62s[0m 432ms/step - loss: 0.1436 - recall: 0.9409 - val_loss: 0.5680 - val_recall: 0.8352 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.1110 - recall: 0.9563 - val_loss: 0.7335 - val_recall: 0.8344 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 380ms/step - loss: 0.1024 - recall: 0.9607 - val_loss: 0.9613 - val_recall: 0.8291 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 374ms/step - loss: 0.1047 - recall: 0.9615 - val_loss: 0.6784 - val_recall: 0.8475 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 377ms/step - loss: 0.0612 - recall: 0.9779 - val_loss: 0.7055 - val_recall: 0.8580 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.0465 - recall: 0.9825 - val_loss: 0.7504 - val_recall: 0.8615 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 376ms/step - loss: 0.0426 - recall: 0.9825 - val_loss: 0.8035 - val_recall: 0.8624 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 379ms/step - loss: 0.0373 - recall: 0.9885 - val_loss: 0.7971 - val_recall: 0.8624 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 377ms/step - loss: 0.0427 - recall: 0.9842 - val_loss: 0.7896 - val_recall: 0.8606 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 377ms/step - loss: 0.0323 - recall: 0.9903 - val_loss: 0.7911 - val_recall: 0.8606 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 378ms/step - loss: 0.0418 - recall: 0.9818 - val_loss: 0.7901 - val_recall: 0.8606 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with dropout and batch normalization regularization
# on the independent validation set
##################################
model_cdrbnr_complex_y_pred_val = model_cdrbnr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 131ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with dropout and batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_cdrbnr_complex_history, 'Complex CNN With Dropout and Batch Normalization Regularization : ')

```


    
![png](output_209_0.png)
    



```python
##################################
# Consolidating the predictions
# for a complex CNN with dropout and batch normalization regularization
# on the validation set
##################################
model_cdrbnr_complex_predictions_val = np.array(list(map(lambda x: np.argmax(x), model_cdrbnr_complex_y_pred_val)))
model_cdrbnr_complex_y_true_val = val_gen.classes

##################################
# Formulating the confusion matrix
# for a complex CNN with dropout and batch normalization regularization
# on the validation set
##################################
cmatrix_val = pd.DataFrame(confusion_matrix(model_cdrbnr_complex_y_true_val, model_cdrbnr_complex_predictions_val), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_val, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Complex CNN With Dropout and Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()

```


    
![png](output_210_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a complex CNN with dropout and batch normalization regularization
# for the entire validation set
##################################
model_cdrbnr_complex_acc_val = accuracy_score(model_cdrbnr_complex_y_true_val, model_cdrbnr_complex_predictions_val)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for the entire validation set
##################################
model_cdrbnr_complex_results_all_val = precision_recall_fscore_support(model_cdrbnr_complex_y_true_val, model_cdrbnr_complex_predictions_val, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for each category of the validation set
##################################
model_cdrbnr_complex_results_class_val = precision_recall_fscore_support(model_cdrbnr_complex_y_true_val, model_cdrbnr_complex_predictions_val, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with dropout and batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_cdrbnr_complex_all_df_val = pd.concat([pd.DataFrame(list(model_cdrbnr_complex_results_class_val)).T,pd.DataFrame(list(model_cdrbnr_complex_results_all_val)).T])
model_cdrbnr_complex_all_df_val.columns = metric_columns
model_cdrbnr_complex_all_df_val.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Complex CNN With Dropout and Batch Normalization Regularization : Validation Set Classification Performance')
model_cdrbnr_complex_all_df_val

```

    Complex CNN With Dropout and Batch Normalization Regularization : Validation Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.848765</td>
      <td>0.862069</td>
      <td>0.855365</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.923077</td>
      <td>0.818182</td>
      <td>0.867470</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.753968</td>
      <td>0.711610</td>
      <td>0.732177</td>
      <td>267.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.845921</td>
      <td>0.962199</td>
      <td>0.900322</td>
      <td>291.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.842933</td>
      <td>0.838515</td>
      <td>0.838834</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_complex_model_list_val = []
model_cdrbnr_complex_measure_list_val = []
model_cdrbnr_complex_category_list_val = []
model_cdrbnr_complex_value_list_val = []
model_cdrbnr_complex_dataset_list_val = []

for i in range(3): 
    for j in range(5):
        model_cdrbnr_complex_model_list_val.append('CNN_CDRBNR_Complex')
        model_cdrbnr_complex_measure_list_val.append(metric_columns[i])
        model_cdrbnr_complex_category_list_val.append(model_cdrbnr_complex_all_df_val.index[j])
        model_cdrbnr_complex_value_list_val.append(model_cdrbnr_complex_all_df_val.iloc[j,i])
        model_cdrbnr_complex_dataset_list_val.append('Validation')

model_cdrbnr_complex_all_summary_val = pd.DataFrame(zip(model_cdrbnr_complex_model_list_val,
                                                        model_cdrbnr_complex_measure_list_val,
                                                        model_cdrbnr_complex_category_list_val,
                                                        model_cdrbnr_complex_value_list_val,
                                                        model_cdrbnr_complex_dataset_list_val), 
                                                    columns=['CNN.Model.Name',
                                                             'Model.Metric',
                                                             'Image.Category',
                                                             'Metric.Value',
                                                             'Data.Set'])
```

### 1.6.7 Model Selection <a class="anchor" id="1.6.7"></a>


```python
##################################
# Consolidating all the
# CNN model performance measures
##################################
cnn_model_performance_comparison_val = pd.concat([model_nr_simple_all_summary_val,
                                                  model_nr_complex_all_summary_val,
                                                  model_dr_simple_all_summary_val,
                                                  model_dr_complex_all_summary_val,
                                                  model_bnr_simple_all_summary_val,
                                                  model_bnr_complex_all_summary_val,
                                                  model_cdrbnr_simple_all_summary_val,
                                                  model_cdrbnr_complex_all_summary_val], 
                                                 ignore_index=True)

```


```python
##################################
# Consolidating all the precision
# model performance measures
##################################
cnn_model_performance_comparison_val_precision = cnn_model_performance_comparison_val[cnn_model_performance_comparison_val['Model.Metric']=='Precision']
cnn_model_performance_comparison_val_precision_CNN_NR_Simple = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_NR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_NR_Complex = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_NR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_DR_Simple = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_DR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_DR_Complex = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_DR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_BNR_Simple = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_BNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_BNR_Complex = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_BNR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_CDRBNR_Simple = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_CDRBNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_precision_CNN_CDRBNR_Complex = cnn_model_performance_comparison_val_precision[cnn_model_performance_comparison_val_precision['CNN.Model.Name']=='CNN_CDRBNR_Complex'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the precision
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_precision_plot = pd.DataFrame({'CNN_NR_Simple': cnn_model_performance_comparison_val_precision_CNN_NR_Simple.values,
                                                                    'CNN_NR_Complex': cnn_model_performance_comparison_val_precision_CNN_NR_Complex.values,
                                                                    'CNN_DR_Simple': cnn_model_performance_comparison_val_precision_CNN_DR_Simple.values,
                                                                    'CNN_DR_Complex': cnn_model_performance_comparison_val_precision_CNN_DR_Complex.values,
                                                                    'CNN_BNR_Simple': cnn_model_performance_comparison_val_precision_CNN_BNR_Simple.values,
                                                                    'CNN_BNR_Complex': cnn_model_performance_comparison_val_precision_CNN_BNR_Complex.values,
                                                                    'CNN_CDRBNR_Simple': cnn_model_performance_comparison_val_precision_CNN_CDRBNR_Simple.values,
                                                                    'CNN_CDRBNR_Complex': cnn_model_performance_comparison_val_precision_CNN_CDRBNR_Complex.values},
                                                                   index=cnn_model_performance_comparison_val_precision['Image.Category'].unique())
cnn_model_performance_comparison_val_precision_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_NR_Simple</th>
      <th>CNN_NR_Complex</th>
      <th>CNN_DR_Simple</th>
      <th>CNN_DR_Complex</th>
      <th>CNN_BNR_Simple</th>
      <th>CNN_BNR_Complex</th>
      <th>CNN_CDRBNR_Simple</th>
      <th>CNN_CDRBNR_Complex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.893238</td>
      <td>0.863057</td>
      <td>0.852459</td>
      <td>0.835913</td>
      <td>0.832317</td>
      <td>0.833333</td>
      <td>0.722656</td>
      <td>0.848765</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.928571</td>
      <td>0.871486</td>
      <td>0.902778</td>
      <td>0.858238</td>
      <td>0.962025</td>
      <td>0.623684</td>
      <td>0.875000</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.624573</td>
      <td>0.655602</td>
      <td>0.554054</td>
      <td>0.641975</td>
      <td>0.690647</td>
      <td>0.430070</td>
      <td>0.313199</td>
      <td>0.753968</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.772595</td>
      <td>0.795252</td>
      <td>0.774691</td>
      <td>0.815287</td>
      <td>0.845638</td>
      <td>0.780083</td>
      <td>0.490698</td>
      <td>0.845921</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.804744</td>
      <td>0.796349</td>
      <td>0.770996</td>
      <td>0.787853</td>
      <td>0.832657</td>
      <td>0.666793</td>
      <td>0.600388</td>
      <td>0.842933</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the precision
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_precision_plot = cnn_model_performance_comparison_val_precision_plot.plot.barh(figsize=(10, 12), width=0.90)
cnn_model_performance_comparison_val_precision_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_precision_plot.set_title("Model Comparison by Precision Performance on Validation Data")
cnn_model_performance_comparison_val_precision_plot.set_xlabel("Precision Performance")
cnn_model_performance_comparison_val_precision_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_precision_plot.grid(False)
cnn_model_performance_comparison_val_precision_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_precision_plot.containers:
    cnn_model_performance_comparison_val_precision_plot.bar_label(container, fmt='%.5f', padding=+10, color='black', fontweight='bold')
    
```


    
![png](output_217_0.png)
    



```python
##################################
# Consolidating all the recall
# model performance measures
##################################
cnn_model_performance_comparison_val_recall = cnn_model_performance_comparison_val[cnn_model_performance_comparison_val['Model.Metric']=='Recall']
cnn_model_performance_comparison_val_recall_CNN_NR_Simple = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_NR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_NR_Complex = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_NR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_DR_Simple = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_DR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_DR_Complex = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_DR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_BNR_Simple = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_BNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_BNR_Complex = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_BNR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_CDRBNR_Simple = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_CDRBNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_recall_CNN_CDRBNR_Complex = cnn_model_performance_comparison_val_recall[cnn_model_performance_comparison_val_recall['CNN.Model.Name']=='CNN_CDRBNR_Complex'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the recall
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_recall_plot = pd.DataFrame({'CNN_NR_Simple': cnn_model_performance_comparison_val_recall_CNN_NR_Simple.values,
                                                                 'CNN_NR_Complex': cnn_model_performance_comparison_val_recall_CNN_NR_Complex.values,
                                                                 'CNN_DR_Simple': cnn_model_performance_comparison_val_recall_CNN_DR_Simple.values,
                                                                 'CNN_DR_Complex': cnn_model_performance_comparison_val_recall_CNN_DR_Complex.values,
                                                                 'CNN_BNR_Simple': cnn_model_performance_comparison_val_recall_CNN_BNR_Simple.values,
                                                                 'CNN_BNR_Complex': cnn_model_performance_comparison_val_recall_CNN_BNR_Complex.values,
                                                                 'CNN_CDRBNR_Simple': cnn_model_performance_comparison_val_recall_CNN_CDRBNR_Simple.values,
                                                                 'CNN_CDRBNR_Complex': cnn_model_performance_comparison_val_recall_CNN_CDRBNR_Complex.values},
                                                                index=cnn_model_performance_comparison_val_recall['Image.Category'].unique())
cnn_model_performance_comparison_val_recall_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_NR_Simple</th>
      <th>CNN_NR_Complex</th>
      <th>CNN_DR_Simple</th>
      <th>CNN_DR_Complex</th>
      <th>CNN_BNR_Simple</th>
      <th>CNN_BNR_Complex</th>
      <th>CNN_CDRBNR_Simple</th>
      <th>CNN_CDRBNR_Complex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.786834</td>
      <td>0.849530</td>
      <td>0.815047</td>
      <td>0.846395</td>
      <td>0.855799</td>
      <td>0.611285</td>
      <td>0.579937</td>
      <td>0.862069</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.787879</td>
      <td>0.821970</td>
      <td>0.738636</td>
      <td>0.848485</td>
      <td>0.863636</td>
      <td>0.897727</td>
      <td>0.026515</td>
      <td>0.818182</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.685393</td>
      <td>0.591760</td>
      <td>0.614232</td>
      <td>0.584270</td>
      <td>0.719101</td>
      <td>0.460674</td>
      <td>0.524345</td>
      <td>0.711610</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.910653</td>
      <td>0.920962</td>
      <td>0.862543</td>
      <td>0.879725</td>
      <td>0.865979</td>
      <td>0.646048</td>
      <td>0.725086</td>
      <td>0.962199</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.792690</td>
      <td>0.796055</td>
      <td>0.757615</td>
      <td>0.789719</td>
      <td>0.826129</td>
      <td>0.653934</td>
      <td>0.463971</td>
      <td>0.838515</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the recall
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_recall_plot = cnn_model_performance_comparison_val_recall_plot.plot.barh(figsize=(10, 12), width=0.90)
cnn_model_performance_comparison_val_recall_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_recall_plot.set_title("Model Comparison by Recall Performance on Validation Data")
cnn_model_performance_comparison_val_recall_plot.set_xlabel("Recall Performance")
cnn_model_performance_comparison_val_recall_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_recall_plot.grid(False)
cnn_model_performance_comparison_val_recall_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_recall_plot.containers:
    cnn_model_performance_comparison_val_recall_plot.bar_label(container, fmt='%.5f', padding=+10, color='black', fontweight='bold')

```


    
![png](output_220_0.png)
    



```python
##################################
# Consolidating all the fscore
# model performance measures
##################################
cnn_model_performance_comparison_val_fscore = cnn_model_performance_comparison_val[cnn_model_performance_comparison_val['Model.Metric']=='F-Score']
cnn_model_performance_comparison_val_fscore_CNN_NR_Simple = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_NR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_NR_Complex = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_NR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_DR_Simple = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_DR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_DR_Complex = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_DR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_BNR_Simple = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_BNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_BNR_Complex = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_BNR_Complex'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_CDRBNR_Simple = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_CDRBNR_Simple'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_fscore_CNN_CDRBNR_Complex = cnn_model_performance_comparison_val_fscore[cnn_model_performance_comparison_val_fscore['CNN.Model.Name']=='CNN_CDRBNR_Complex'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the fscore
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_fscore_plot = pd.DataFrame({'CNN_NR_Simple': cnn_model_performance_comparison_val_fscore_CNN_NR_Simple.values,
                                                                 'CNN_NR_Complex': cnn_model_performance_comparison_val_fscore_CNN_NR_Complex.values,
                                                                 'CNN_DR_Simple': cnn_model_performance_comparison_val_fscore_CNN_DR_Simple.values,
                                                                 'CNN_DR_Complex': cnn_model_performance_comparison_val_fscore_CNN_DR_Complex.values,
                                                                 'CNN_BNR_Simple': cnn_model_performance_comparison_val_fscore_CNN_BNR_Simple.values,
                                                                 'CNN_BNR_Complex': cnn_model_performance_comparison_val_fscore_CNN_BNR_Complex.values,
                                                                 'CNN_CDRBNR_Simple': cnn_model_performance_comparison_val_fscore_CNN_CDRBNR_Simple.values,
                                                                 'CNN_CDRBNR_Complex': cnn_model_performance_comparison_val_fscore_CNN_CDRBNR_Complex.values},
                                                                index=cnn_model_performance_comparison_val_fscore['Image.Category'].unique())
cnn_model_performance_comparison_val_fscore_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_NR_Simple</th>
      <th>CNN_NR_Complex</th>
      <th>CNN_DR_Simple</th>
      <th>CNN_DR_Complex</th>
      <th>CNN_BNR_Simple</th>
      <th>CNN_BNR_Complex</th>
      <th>CNN_CDRBNR_Simple</th>
      <th>CNN_CDRBNR_Complex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.836667</td>
      <td>0.856240</td>
      <td>0.833333</td>
      <td>0.841121</td>
      <td>0.843895</td>
      <td>0.705244</td>
      <td>0.643478</td>
      <td>0.855365</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.852459</td>
      <td>0.846004</td>
      <td>0.812500</td>
      <td>0.853333</td>
      <td>0.910180</td>
      <td>0.736025</td>
      <td>0.051471</td>
      <td>0.867470</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.653571</td>
      <td>0.622047</td>
      <td>0.582593</td>
      <td>0.611765</td>
      <td>0.704587</td>
      <td>0.444846</td>
      <td>0.392157</td>
      <td>0.732177</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.835962</td>
      <td>0.853503</td>
      <td>0.816260</td>
      <td>0.846281</td>
      <td>0.855688</td>
      <td>0.706767</td>
      <td>0.585298</td>
      <td>0.900322</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.794665</td>
      <td>0.794449</td>
      <td>0.761172</td>
      <td>0.788125</td>
      <td>0.828587</td>
      <td>0.648221</td>
      <td>0.418101</td>
      <td>0.838834</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the fscore
# model performance measures
# for all CNN models
##################################
cnn_model_performance_comparison_val_fscore_plot = cnn_model_performance_comparison_val_fscore_plot.plot.barh(figsize=(10, 12), width=0.90)
cnn_model_performance_comparison_val_fscore_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_fscore_plot.set_title("Model Comparison by F-Score Performance on Validation Data")
cnn_model_performance_comparison_val_fscore_plot.set_xlabel("F-Score Performance")
cnn_model_performance_comparison_val_fscore_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_fscore_plot.grid(False)
cnn_model_performance_comparison_val_fscore_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_fscore_plot.containers:
    cnn_model_performance_comparison_val_fscore_plot.bar_label(container, fmt='%.5f', padding=+10, color='black', fontweight='bold')

```


    
![png](output_223_0.png)
    


### 1.6.8 Model Testing <a class="anchor" id="1.6.9"></a>


```python
##################################
# Evaluating the model
# for a complex CNN with dropout and batch normalization regularization
# on the independent test set
##################################
model_cdrbnr_complex_y_pred_test = model_cdrbnr_complex.predict(test_gen)
```

    [1m41/41[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 108ms/step
    


```python
##################################
# Consolidating the predictions
# for a complex CNN with dropout and batch normalization regularization
# on the test set
##################################
model_cdrbnr_complex_predictions_test = np.array(list(map(lambda x: np.argmax(x), model_cdrbnr_complex_y_pred_test)))
model_cdrbnr_complex_y_true_test = test_gen.classes

##################################
# Formulating the confusion matrix
# for a complex CNN with dropout and batch normalization regularization
# on the test set
##################################
cmatrix_test = pd.DataFrame(confusion_matrix(model_cdrbnr_complex_y_true_test, model_cdrbnr_complex_predictions_test), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for each category of the test set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_test, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Complex CNN With Dropout and Batch Normalization Regularization : Test Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

```


    
![png](output_226_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a complex CNN with dropout and batch normalization regularization
# for the entire test set
##################################
model_cdrbnr_complex_acc_test = accuracy_score(model_cdrbnr_complex_y_true_test, model_cdrbnr_complex_predictions_test)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for the entire test set
##################################
model_cdrbnr_complex_results_all_test = precision_recall_fscore_support(model_cdrbnr_complex_y_true_test, model_cdrbnr_complex_predictions_test, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a complex CNN with dropout and batch normalization regularization
# for each category of the test set
##################################
model_cdrbnr_complex_results_class_test = precision_recall_fscore_support(model_cdrbnr_complex_y_true_test, model_cdrbnr_complex_predictions_test, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a complex CNN with dropout and batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_cdrbnr_complex_all_df_test = pd.concat([pd.DataFrame(list(model_cdrbnr_complex_results_class_test)).T,pd.DataFrame(list(model_cdrbnr_complex_results_all_test)).T])
model_cdrbnr_complex_all_df_test.columns = metric_columns
model_cdrbnr_complex_all_df_test.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Complex CNN With Dropout and Batch Normalization Regularization : Test Set Classification Performance')
model_cdrbnr_complex_all_df_test

```

    Complex CNN With Dropout and Batch Normalization Regularization : Test Set Classification Performance
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.860215</td>
      <td>0.987654</td>
      <td>0.919540</td>
      <td>405.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.923345</td>
      <td>0.883333</td>
      <td>0.902896</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.864542</td>
      <td>0.709150</td>
      <td>0.779174</td>
      <td>306.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.938312</td>
      <td>0.963333</td>
      <td>0.950658</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.896603</td>
      <td>0.885868</td>
      <td>0.888067</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_complex_model_list_test = []
model_cdrbnr_complex_measure_list_test = []
model_cdrbnr_complex_category_list_test = []
model_cdrbnr_complex_value_list_test = []
model_cdrbnr_complex_dataset_list_test = []

for i in range(3): 
    for j in range(5):
        model_cdrbnr_complex_model_list_test.append('CNN_CDRBNR_Complex')
        model_cdrbnr_complex_measure_list_test.append(metric_columns[i])
        model_cdrbnr_complex_category_list_test.append(model_cdrbnr_complex_all_df_test.index[j])
        model_cdrbnr_complex_value_list_test.append(model_cdrbnr_complex_all_df_test.iloc[j,i])
        model_cdrbnr_complex_dataset_list_test.append('Test')

model_cdrbnr_complex_all_summary_test = pd.DataFrame(zip(model_cdrbnr_complex_model_list_test,
                                                         model_cdrbnr_complex_measure_list_test,
                                                         model_cdrbnr_complex_category_list_test,
                                                         model_cdrbnr_complex_value_list_test,
                                                         model_cdrbnr_complex_dataset_list_test), 
                                                     columns=['CNN.Model.Name',
                                                              'Model.Metric',
                                                              'Image.Category',
                                                              'Metric.Value',
                                                              'Data.Set'])

```


```python
##################################
# Consolidating all the
# CNN model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test = pd.concat([model_cdrbnr_complex_all_summary_val,
                                                       model_cdrbnr_complex_all_summary_test], 
                                                      ignore_index=True)

```


```python
##################################
# Consolidating all the precision
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_precision = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='Precision']
cnn_model_performance_comparison_val_test_precision_CNN_CDRBNR_Complex_validation = cnn_model_performance_comparison_val_test_precision[cnn_model_performance_comparison_val_test_precision['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_precision_CNN_CDRBNR_Complex_test = cnn_model_performance_comparison_val_test_precision[cnn_model_performance_comparison_val_test_precision['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the precision
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_precision_plot = pd.DataFrame({'CNN_CDRBNR_Complex_Validation': cnn_model_performance_comparison_val_test_precision_CNN_CDRBNR_Complex_validation.values,
                                                                         'CNN_CDRBNR_Complex_Test': cnn_model_performance_comparison_val_test_precision_CNN_CDRBNR_Complex_test.values},
                                                                        cnn_model_performance_comparison_val_test_precision['Image.Category'].unique())
cnn_model_performance_comparison_val_test_precision_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_CDRBNR_Complex_Validation</th>
      <th>CNN_CDRBNR_Complex_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.848765</td>
      <td>0.860215</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.923077</td>
      <td>0.923345</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.753968</td>
      <td>0.864542</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.845921</td>
      <td>0.938312</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.842933</td>
      <td>0.896603</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the precision
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_precision_plot = cnn_model_performance_comparison_val_test_precision_plot.plot.barh(figsize=(10, 6), width=0.90)
cnn_model_performance_comparison_val_test_precision_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_test_precision_plot.set_title("Model Precision Performance Comparison on Validation and Test Data")
cnn_model_performance_comparison_val_test_precision_plot.set_xlabel("Precision Performance")
cnn_model_performance_comparison_val_test_precision_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_test_precision_plot.grid(False)
cnn_model_performance_comparison_val_test_precision_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_test_precision_plot.containers:
    cnn_model_performance_comparison_val_test_precision_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')
    
```


    
![png](output_232_0.png)
    



```python
##################################
# Consolidating all the recall
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_recall = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='Recall']
cnn_model_performance_comparison_val_test_recall_CNN_CDRBNR_Complex_validation = cnn_model_performance_comparison_val_test_recall[cnn_model_performance_comparison_val_test_recall['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_recall_CNN_CDRBNR_Complex_test = cnn_model_performance_comparison_val_test_recall[cnn_model_performance_comparison_val_test_recall['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the recall
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_recall_plot = pd.DataFrame({'CNN_CDRBNR_Complex_Validation': cnn_model_performance_comparison_val_test_recall_CNN_CDRBNR_Complex_validation.values,
                                                                      'CNN_CDRBNR_Complex_Test': cnn_model_performance_comparison_val_test_recall_CNN_CDRBNR_Complex_test.values},
                                                                     cnn_model_performance_comparison_val_test_recall['Image.Category'].unique())
cnn_model_performance_comparison_val_test_recall_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_CDRBNR_Complex_Validation</th>
      <th>CNN_CDRBNR_Complex_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.862069</td>
      <td>0.987654</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.818182</td>
      <td>0.883333</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.711610</td>
      <td>0.709150</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.962199</td>
      <td>0.963333</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.838515</td>
      <td>0.885868</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the recall
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_recall_plot = cnn_model_performance_comparison_val_test_recall_plot.plot.barh(figsize=(10, 6), width=0.90)
cnn_model_performance_comparison_val_test_recall_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_test_recall_plot.set_title("Model Recall Performance Comparison on Validation and Test Data")
cnn_model_performance_comparison_val_test_recall_plot.set_xlabel("Recall Performance")
cnn_model_performance_comparison_val_test_recall_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_test_recall_plot.grid(False)
cnn_model_performance_comparison_val_test_recall_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_test_recall_plot.containers:
    cnn_model_performance_comparison_val_test_recall_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')

```


    
![png](output_235_0.png)
    



```python
##################################
# Consolidating all the fscore
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_fscore = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='F-Score']
cnn_model_performance_comparison_val_test_fscore_CNN_CDRBNR_Complex_validation = cnn_model_performance_comparison_val_test_fscore[cnn_model_performance_comparison_val_test_fscore['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_fscore_CNN_CDRBNR_Complex_test = cnn_model_performance_comparison_val_test_fscore[cnn_model_performance_comparison_val_test_fscore['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the fscore
# model performance measures
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_fscore_plot = pd.DataFrame({'CNN_CDRBNR_Complex_Validation': cnn_model_performance_comparison_val_test_fscore_CNN_CDRBNR_Complex_validation.values,
                                                                      'CNN_CDRBNR_Complex_Test': cnn_model_performance_comparison_val_test_fscore_CNN_CDRBNR_Complex_test.values},
                                                                     cnn_model_performance_comparison_val_test_fscore['Image.Category'].unique())
cnn_model_performance_comparison_val_test_fscore_plot

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN_CDRBNR_Complex_Validation</th>
      <th>CNN_CDRBNR_Complex_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.855365</td>
      <td>0.919540</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.867470</td>
      <td>0.902896</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.732177</td>
      <td>0.779174</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.900322</td>
      <td>0.950658</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.838834</td>
      <td>0.888067</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the fscore
# for the selected model
# complex CNN with dropout and batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_fscore_plot = cnn_model_performance_comparison_val_test_fscore_plot.plot.barh(figsize=(10, 6), width=0.90)
cnn_model_performance_comparison_val_test_fscore_plot.set_xlim(-0.02,1.10)
cnn_model_performance_comparison_val_test_fscore_plot.set_title("Model F-Score Performance Comparison on Validation and Test Data")
cnn_model_performance_comparison_val_test_fscore_plot.set_xlabel("F-Score Performance")
cnn_model_performance_comparison_val_test_fscore_plot.set_ylabel("Image Categories")
cnn_model_performance_comparison_val_test_fscore_plot.grid(False)
cnn_model_performance_comparison_val_test_fscore_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in cnn_model_performance_comparison_val_test_fscore_plot.containers:
    cnn_model_performance_comparison_val_test_fscore_plot.bar_label(container, fmt='%.5f', padding=-50, color='white', fontweight='bold')

```


    
![png](output_238_0.png)
    


### 1.6.9 Model Inference <a class="anchor" id="1.6.10"></a>

## 1.7 Predictive Model Development <a class="anchor" id="1.7"></a>

### 1.7.1 Model Application Programming Interface Code Development <a class="anchor" id="1.7.1"></a>

### 1.7.2 User Interface Application Code Development <a class="anchor" id="1.7.2"></a>

### 1.7.3 Web Application <a class="anchor" id="1.7.3"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by Francois Chollet
* **[Book]** [Deep Learning: A Visual Approach](https://nostarch.com/deep-learning-visual-approach) by Andrew Glassner
* **[Book]** [Learning Deep Learning](https://ldlbook.com/) by Magnus Ekman
* **[Book]** [Practical Deep Learning](https://nostarch.com/practical-deep-learning-python) by Ronald Kneusel
* **[Book]** [Deep Learning with Tensorflow and Keras](https://www.packtpub.com/product/deep-learning-with-tensorflow-and-keras-third-edition/9781803232911) by Amita Kapoor, Antonio Gulli and Sujit Pal
* **[Book]** [Deep Learning](https://mitpress.mit.edu/9780262537551/deep-learning/) by John Kelleher
* **[Book]** [Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/) by David Foster
* **[Book]** [Deep Learning Illustrated](https://www.deeplearningillustrated.com/) by John Krohn, Grant Beyleveld and Aglae Bassens
* **[Book]** [Neural Networks and Deep Learning](https://link.springer.com/book/10.1007/978-3-319-94463-0) by Charu Aggarwal
* **[Book]** [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) by Andrew Trask
* **[Book]** [Deep Learning with Pytorch](https://www.manning.com/books/deep-learning-with-pytorch) by Eli Stevens, Luca Antiga and Thomas Viehmann
* **[Book]** [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio and Aaron Courville
* **[Book]** [Deep Learning from Scratch](https://www.oreilly.com/library/view/deep-learning-from/9781492041405/) by Seth Weidman
* **[Book]** [Fundamentals of Deep Learning](https://www.oreilly.com/library/view/fundamentals-of-deep/9781492082170/) by Nithin Buduma, Nikhil Buduma and Joe Papa
* **[Book]** [Hands-On Machine Learning with Scikit-Learn, Keras and Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron
* **[Book]** [Deep Learning for Computer Vision](https://machinelearningmastery.com/deep-learning-for-computer-vision/) by Jason Brownlee
* **[Python Library API]** [numpy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.image](https://matplotlib.org/stable/api/image_api.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.offsetbox](https://matplotlib.org/stable/api/offsetbox_api.html) by MatPlotLib Team
* **[Python Library API]** [tensorflow](https://pypi.org/project/tensorflow/) by TensorFlow Team
* **[Python Library API]** [keras](https://pypi.org/project/keras/) by Keras Team
* **[Python Library API]** [pil](https://pypi.org/project/Pillow/) by Pillow Team
* **[Python Library API]** [glob](https://docs.python.org/3/library/glob.html) by glob Team
* **[Python Library API]** [cv2](https://pypi.org/project/opencv-python/) by OpenCV Team
* **[Python Library API]** [os](https://docs.python.org/3/library/os.html) by os Team
* **[Python Library API]** [random](https://docs.python.org/3/library/random.html) by random Team
* **[Python Library API]** [keras.models](https://www.tensorflow.org/api_docs/python/tf/keras/models) by TensorFlow Team
* **[Python Library API]** [keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) by TensorFlow Team
* **[Python Library API]** [keras.wrappers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Wrapper) by TensorFlow Team
* **[Python Library API]** [keras.utils](https://www.tensorflow.org/api_docs/python/tf/keras/utils) by TensorFlow Team
* **[Python Library API]** [keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) by TensorFlow Team
* **[Python Library API]** [keras.preprocessing.image](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image) by TensorFlow Team
* **[Python Library API]** [keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) by TensorFlow Team
* **[Python Library API]** [keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) by TensorFlow Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Article]** [Convolutional Neural Networks, Explained](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939) by Mayank Mishra (Towards Data Science)
* **[Article]** [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) by Sumit Saha (Towards Data Science)
* **[Article]** [Understanding Convolutional Neural Networks: A Beginner’s Journey into the Architecture](https://medium.com/codex/understanding-convolutional-neural-networks-a-beginners-journey-into-the-architecture-aab30dface10) by Afaque Umer (Medium)
* **[Article]** [Introduction to Convolutional Neural Networks (CNN)](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/) by Manav Mandal (Analytics Vidhya)
* **[Article]** [What Are Convolutional Neural Networks?](https://www.ibm.com/topics/convolutional-neural-networks) by IBM Team (IBM)
* **[Article]** [What is CNN? A 5 Year Old guide to Convolutional Neural Network](https://medium.com/analytics-vidhya/what-is-cnn-a-5-year-old-guide-to-convolutional-neural-network-c9d0338c3bf7) by William Ong (Medium)
* **[Article]** [Convolutional Neural Network](https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network) by Thomas Wood (DeepAI.Org)
* **[Article]** [How Do Convolutional Layers Work in Deep Learning Neural Networks?](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Convolutional Neural Networks Explained: Using PyTorch to Understand CNNs](https://builtin.com/data-science/convolutional-neural-networks-explained) by Vihar Kurama (BuiltIn)
* **[Article]** [Convolutional Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks) by Afshine Amidi and Shervine Amidi (Stanford University)
* **[Article]** [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) by Ujjwal Karn (The Data Science Blog)
* **[Article]** [Convolutional Neural Network](https://www.nvidia.com/en-us/glossary/data-science/convolutional-neural-network/) by NVIDIA Team (NVIDIA)
* **[Article]** [Convolutional Neural Networks (CNN) Overview](https://encord.com/blog/convolutional-neural-networks-explained/) by Nikolaj Buhl (Encord)
* **[Article]** [Understanding Convolutional Neural Network (CNN): A Complete Guide](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/) by LearnOpenCV Team (LearnOpenCV)
* **[Article]** [Convolutional Neural Networks (CNNs) and Layer Types](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/) by Adrian Rosebrock (PyImageSearch)
* **[Article]** [How Convolutional Neural Networks See The World](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html) by Francois Chollet (The Keras Blog)
* **[Article]** [What Is a Convolutional Neural Network?](https://www.mathworks.com/discovery/convolutional-neural-network-matlab.html#:~:text=A%20convolutional%20neural%20network%20(CNN,%2Dseries%2C%20and%20signal%20data.) by MathWorks Team (MathWorks)
* **[Article]** [Grad-CAM Class Activation Visualization](https://keras.io/examples/vision/grad_cam/) by Francois Chollet (Keras.IO)
* **[Article]** [Grad-CAM: Visualize Class Activation Maps with Keras, TensorFlow, and Deep Learning](https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock (PyImageSearch)
* **[Kaggle Project]** [glioma 19 Radiography Data - EDA and CNN Model](https://www.kaggle.com/code/jnegrini/glioma-19-radiography-data-eda-and-cnn-model) by Juliana Negrini De Araujo (Kaggle)
* **[Kaggle Project]** [Pneumonia Detection using CNN (92.6% Accuracy)](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy) by Madhav Mathur (Kaggle)
* **[Kaggle Project]** [glioma Detection from CXR Using Explainable CNN](https://www.kaggle.com/code/sid321axn/glioma-detection-from-cxr-using-explainable-cnn) by Manu Siddhartha (Kaggle)
* **[Kaggle Project]** [Class Activation Mapping for glioma-19 CNN](https://www.kaggle.com/code/amyjang/class-activation-mapping-for-glioma-19-cnn) by Amy Zhang (Kaggle)
* **[Kaggle Project]** [CNN mri glioma Classification](https://www.kaggle.com/code/gabrielmino/cnn-mri-glioma-classification) by Gabriel Mino (Kaggle)
* **[Kaggle Project]** [Detecting-glioma-19-Images | CNN](https://www.kaggle.com/code/felipeoliveiraml/detecting-glioma-19-images-cnn) by Felipe Oliveira (Kaggle)
* **[Kaggle Project]** [Detection of glioma Positive Cases using DL](https://www.kaggle.com/code/sana306/detection-of-glioma-positive-cases-using-dl) by Sana Shaikh (Kaggle)
* **[Kaggle Project]** [Deep Learning and Transfer Learning on glioma-19](https://www.kaggle.com/code/digvijayyadav/deep-learning-and-transfer-learning-on-glioma-19) by Digvijay Yadav (Kaggle)
* **[Kaggle Project]** [X-ray Detecting Using CNN](https://www.kaggle.com/code/shivan118/x-ray-detecting-using-cnn) by Shivan Kumar (Kaggle)
* **[Kaggle Project]** [Classification of glioma-19 using CNN](https://www.kaggle.com/code/islamselim/classification-of-glioma-19-using-cnn) by Islam Selim (Kaggle)
* **[Kaggle Project]** [glioma-19 - Revisiting Pneumonia Detection](https://www.kaggle.com/code/pcbreviglieri/glioma-19-revisiting-pneumonia-detection) by Paulo Breviglieri (Kaggle)
* **[Kaggle Project]** [Multi-Class X-ray glioma19 Classification-94% Accurary](https://www.kaggle.com/code/derrelldsouza/multi-class-x-ray-glioma19-classification-94-acc) by Quadeer Shaikh (Kaggle)
* **[Kaggle Project]** [Grad-CAM: What Do CNNs See?](https://www.kaggle.com/code/quadeer15sh/grad-cam-what-do-cnns-see) by Derrel Souza (Kaggle)
* **[GitHub Project]** [Grad-CAM](https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb) by Ismail Uddin (GitHub)
* **[Publication]** [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) by Yann LeCun, Leon Bottou, Yoshua Bengio and Patrick Haffner (Proceedings of the IEEE)
* **[Publication]** [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) by Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva and Antonio Torralba (Computer Vision and Pattern Recognition)
* **[Publication]** [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) by Ramprasaath Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh and Dhruv Batra (Computer Vision and Pattern Recognition)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

