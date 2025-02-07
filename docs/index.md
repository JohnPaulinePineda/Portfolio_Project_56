***
# Model Deployment : Classifying Brain Tumors from Magnetic Resonance Images by Leveraging Convolutional Neural Network-Based Multilevel Feature Extraction and Hierarchical Representation 

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *January 3, 2025*
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

This project focuses on leveraging the **Convolutional Neural Network Model** using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> for multiclass image classification by directly learning hierarchical features from raw pixel data. The CNN models were designed to extract low- and high-level features for differentiating between image categories. Various hyperparameters, including the number of layers, filter size, and number of dense layer weights, were systematically evaluated to optimize the model architecture. **Image Augmentation** techniques were employed to increase the diversity of training images and improve the model's ability to generalize. To enhance model performance and robustness, various regularization techniques were explored, including **Dropout**, **Batch Normalization**, and their **Combinations**. These methods helped mitigate overfitting and ensured stable learning. Callback functions such as **Early Stopping**, **Learning Rate Reduction on Performance Plateaus**, and **Model Checkpointing** were implemented to fine-tune the training process, optimize convergence, and prevent overtraining. Model evaluation was conducted using **Precision**, **Recall**, and **F1 Score** metrics to ensure both false positives and false negatives are considered, providing a more balanced view of model classification performance. Post-training, interpretability was emphasized through an advanced visualization technique using **Gradient Class Activation Mapping (Grad-CAM)**,  providing insights into the spatial and hierarchical features that influenced the model's predictions, offering a deeper understanding of the decision-making process. The final model was deployed as a prototype application with a web interface via **Streamlit**, enabling interactive exploration of predictions and model results. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Machine Learning Classification Models](https://nostarch.com/deep-learning-visual-approach) are algorithms that learn to assign predefined categories or labels to input data based on patterns and relationships identified during the training phase. Classification is a supervised learning task, meaning the models are trained on a labeled dataset where the correct output (class or label) is known for each input. Once trained, these models can predict the class of new, unseen instances.

[Convolutional Neural Network Models](https://www.manning.com/books/deep-learning-with-python-second-edition) are a neural network architecture specifically designed for image classification and computer vision tasks by automatically learning hierarchical features directly from raw pixel data. The core building block of a CNN is the convolutional layer. Convolution operations apply learnable filters (kernels) to input images to detect patterns such as edges, textures, and more complex structures. The layers systematically learn hierarchical features from low-level (e.g., edges) to high-level (e.g., object parts) as the network deepens. Filters are shared across the entire input space, enabling the model to recognize patterns regardless of their spatial location. After convolutional operations, an activation function is applied element-wise to introduce non-linearity and allow the model to learn complex relationships between features. Pooling layers downsample the spatial dimensions of the feature maps, reducing the computational load and the number of parameters in the network - creating spatial hierarchy and translation invariance. Fully connected layers process the flattened features to make predictions and produce an output vector that corresponds to class probabilities using an activation function. The CNN is trained using backpropagation and optimization algorithms. A loss function is used to measure the difference between predicted and actual labels. The network adjusts its weights to minimize this loss. Gradients are calculated with respect to the loss, and the weights are updated accordingly through a backpropagation mechanism.

[Regularization Methods](https://www.manning.com/books/deep-learning-with-python-second-edition) in deep learning, particularly in CNNs for image classification, refers to techniques used to reduce overfitting and improve generalization. Overfitting occurs when a model learns not only the underlying patterns but also the noise in the training data, leading to poor performance on unseen data. CNNs are prone to overfitting due to their high capacity and ability to memorize training data. Regularization techniques work by limiting the network's ability to overly specialize, ensuring it generalizes well to unseen data while still retaining its ability to learn important patterns from the input images.

[Dropout Regularization](https://www.manning.com/books/deep-learning-with-python-second-edition) randomly sets a fraction of activations to zero during training, forcing the network to learn redundant representations. When a CNN is too complex or has too many parameters (weights), it might memorize the training data instead of learning the patterns. This leads to overfitting, where the model performs well on training data but poorly on new, unseen data. Dropout forces the network to rely on multiple neurons rather than depending on just a few. It does this by randomly turning off (setting to zero) a fraction of neurons during each training iteration. This process of turning off means the model cannot rely too heavily on any single feature or path through the network. During training, each unit (neuron) has a defined probability of being dropped. During inference, all units are used but scaled by this probability subtracted from one. Dropout layers are typically used after fully connected layers. Dropout is not used in convolutional layers as often because the spatial structure is critical for convolutional operations. However, it is effective after pooling or fully connected layers, which aggregate features and are prone to overfitting.

[Batch Normalization Regularization](https://www.manning.com/books/deep-learning-with-python-second-edition) normalizes the input to a layer within a mini-batch to have a mean of zero and standard deviation of one, and introduces learnable scaling and shifting parameters. During training, as data flows through the layers of a CNN, the scale of the data can change unpredictably. This can cause vanishing gradients (tiny gradients make learning very slow or even stop), exploding gradients (huge gradients destabilize training, causing weights to grow excessively) and slow convergence (training takes much longer because the optimizer struggles to find a good direction to minimize the loss). Batch normalization stabilizes the learning process by normalizing the data at each layer. It adjusts the outputs of each layer so they have a consistent mean (centered at 0) and standard deviation (spread out to 1) across a mini-batch of data. Additionally, batch normalization introduces two trainable parameters (scale and shift), allowing the network to adapt if normalization disrupts the patterns the network is trying to learn. Batch normalization layers are typically applied between the linear transformation and activation function in layers. Batch normalization stabilizes the learning process and prevents gradients from vanishing or exploding, which is particularly beneficial in deep CNNs.

[Convolutional Layer Filter Visualization](https://www.manning.com/books/deep-learning-with-python-second-edition) helps in understanding what specific patterns or features the CNN has learned during the training process. Given that convolutional layers learn filters act as feature extractors, visualizing these filters can provide insights into the types of patterns or textures the network is sensitive to. In addition, image representations of filters allows the assessment of how the complexity of features evolve through the network with low-level features such as edges or textures captured in the earlier layers, while filters in deeper layers detecting more abstract and complex features. By applying learned filters to an input image, it is possible to visualize which regions of the image activate specific filters the most. This can aid in identifying which parts of the input contribute most to the response of a particular filter, providing insights into what the network focuses on.

[Gradient-Weighted Class Activation Maps](https://www.manning.com/books/deep-learning-with-python-second-edition) highlight the regions of an input image that contribute the most to a specific class prediction from a CNN model by providing a heatmap that indicates the importance of different regions in the input image for a particular classification decision. Grad-CAM helps identify which regions of the input image are crucial for a CNN's decision on a specific class. It provides a localization map that highlights the relevant parts of the image that contribute to the predicted class. By overlaying the Grad-CAM heatmap on the original image, one can visually understand where the model is focusing its attention when making predictions. This spatial understanding is particularly valuable for tasks such as object detection or segmentation.

[Streamlit](https://streamlit.io/) is an open-source Python library that simplifies the creation and deployment of web applications for machine learning and data science projects. It allows developers and data scientists to turn Python scripts into interactive web apps quickly without requiring extensive web development knowledge. Streamlit seamlessly integrates with popular Python libraries such as Pandas, Matplotlib, Plotly, and TensorFlow, allowing one to leverage existing data processing and visualization tools within the application. Streamlit apps can be easily deployed on various platforms, including Streamlit Community Cloud, Heroku, or any cloud service that supports Python web applications.

[Streamlit Community Cloud](https://streamlit.io/cloud), formerly known as Streamlit Sharing, is a free cloud-based platform provided by Streamlit that allows users to easily deploy and share Streamlit apps online. It is particularly popular among data scientists, machine learning engineers, and developers for quickly showcasing projects, creating interactive demos, and sharing data-driven applications with a wider audience without needing to manage server infrastructure. Significant features include free hosting (Streamlit Community Cloud provides free hosting for Streamlit apps, making it accessible for users who want to share their work without incurring hosting costs), easy deployment (users can connect their GitHub repository to Streamlit Community Cloud, and the app is automatically deployed from the repository), continuous deployment (if the code in the connected GitHub repository is updated, the app is automatically redeployed with the latest changes), 
sharing capabilities (once deployed, apps can be shared with others via a simple URL, making it easy for collaborators, stakeholders, or the general public to access and interact with the app), built-in authentication (users can restrict access to their apps using GitHub-based authentication, allowing control over who can view and interact with the app), and community support (the platform is supported by a community of users and developers who share knowledge, templates, and best practices for building and deploying Streamlit apps).


## 1.1 Data Background <a class="anchor" id="1.1"></a>

An open [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar)) was used for the analysis as consolidated from the following primary sources: 
1. [Research Data Repository](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) entitled **Brain Tumor Dataset** by [Jun Cheng](https://figshare.com/authors/Jun_Cheng/784074) from [FigShare](https://figshare.com/)
2. [Research Data Repository](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) entitled **Brain Tumor Classification (MRI)** by [Sartaj Bhuvaji](https://www.kaggle.com/sartajbhuvaji) from [Kaggle](https://www.kaggle.com/)
3. [GitHub Code Repository](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms) entitled **Brain Tumor Classification Using Deep Learning Algorithms** by [Sartaj Bhuvaji](https://github.com/SartajBhuvaji) from [GitHub](https://github.com/)
4. [Research Data Repository](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) entitled **Br35H :: Brain Tumor Detection 2020** by [Ahmed Hamada](https://www.kaggle.com/ahmedhamada0) from [Kaggle](https://www.kaggle.com/)

This study hypothesized that images contain a hierarchy of features which allows the differentiation and classification across various image categories.

The multiclass categorical variable for the study is:
* <span style="color: #FF0000">CLASS</span> - Multi-categorical diagnostic classification for the brain MRI (No Tumor | Glioma | Meningioma | Pituitary)

The hierarchical representation of image features enables the network to transform raw pixel data into a meaningful and compact representation, allowing it to make accurate predictions during image classification. The different features automatically learned during the training process are as follows::
* <span style="color: #FF0000">OBJECT TEXTURE</span> - Repetitive gradients among pixel intensities
* <span style="color: #FF0000">OBJECT EDGE</span> - Abrupt changes or transitions on pixel intensities
* <span style="color: #FF0000">OBJECT PATTERN</span> - Distinctive structural features in pixel intensities
* <span style="color: #FF0000">OBJECT SHAPE</span> - Spatial relationships and contours among pixel intensities
* <span style="color: #FF0000">SPATIAL HIERARCHY</span> - Layered abstract representations of spatial structures in image objects
* <span style="color: #FF0000">SPATIAL LOCALIZATION</span> - Boundaries and position of the object within the image
 

## 1.2 Data Description <a class="anchor" id="1.2"></a>

1. The (initial) training dataset is comprised of:
    * **5712 images** (observations)
    * **1 target** (variable)
        * <span style="color: #FF0000">CLASS: No Tumor</span> = **1595 images**
        * <span style="color: #FF0000">CLASS: Pituitary</span> = **1457 images**
        * <span style="color: #FF0000">CLASS: Meningioma</span> = **1339 images**
        * <span style="color: #FF0000">CLASS: Glioma</span> = **1321 images**
2. The test dataset is comprised of:
    * **1311 images** (observations)
    * **1 target** (variable)
        * <span style="color: #FF0000">CLASS: No Tumor</span> = **405 images**
        * <span style="color: #FF0000">CLASS: Pituitary</span> = **306 images**
        * <span style="color: #FF0000">CLASS: Meningioma</span> = **300 images**
        * <span style="color: #FF0000">CLASS: Glioma</span> = **300 images**



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
from collections import Counter
import scipy.stats as stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

##################################
# Model Development
##################################
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
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
MODELS_PATH = r"models\candidates"
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
# for the testing data
##################################
diagnosis_code_dictionary_test = {'Te-no': 0,
                                  'Te-noTr': 0,
                                  'Te-gl': 1,
                                  'Te-glTr': 1,
                                  'Te-me': 2,
                                  'Te-meTr': 2,
                                  'Te-pi': 3,
                                  'Te-piTr': 3}

##################################
# Defining the image category descriptions
# for the testing data
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
# from the testing data
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
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
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
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Te-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Te-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Te-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Te-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
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
      <td>1311.0</td>
      <td>1.382151</td>
      <td>1.1457</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
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

Data quality findings based on assessment are as follows:
1. No duplicated images observed.
2. No null images observed.


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




    Image_ID     0
    Path         0
    Diagnosis    0
    Target       0
    Class        0
    dtype: int64



## 1.4 Data Preprocessing <a class="anchor" id="1.4"></a>

1. Each grayscale image contains 3 channels with equivalent pixel values for each individual channel:
    * Red channel pixel value range = 0 to 255
    * Blue channel pixel value range = 0 to 255
    * Green channel pixel value range = 0 to 255
2. All images were resized to a consistent shape, allowing them to be processed in batches by the model and to work seamlessly with augmentation techniques ensuring consistency in image preprocessing.
    * Image height = 227 pixels
    * Image width = 227 pixels
    * Image size = 51,529 pixels
3. Different image augmentation techniques were applied using various transformations to the training images to artificially increase the diversity of the dataset and improve the generalization and robustness of the model, including:
    * **Rescaling** - normalization of the pixel values within the 0 to 1 range
    * **Rotation** - random image rotation by 2 degrees
    * **Width Shift** - random horizontal shifting of the image by 2% of the total width
    * **Height Shift** - random vertical shifting of the image by 2% of the total height
    * **Shear Transformation** - image slanting by 2 degrees along the horizontal axis.
    * **Zooming** - random image zoom-in or zoom-out by a factor of 2%
4. Other image augmentation techniques were not applied to minimize noise in the dataset, including:
    * **Horizontal Flip** - random horizontal flipping of the image
    * **Vertical Flip** - random vertical flipping of the image



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
# Obtaining the count and class distribution
# for the training set
##################################
total_images = train_gen.samples
class_counts = Counter(train_gen.classes)
print("Training Set Class Breakdown:")
print(f"Overall: {total_images}")
for class_id, count in class_counts.items():
    class_name = list(train_gen.class_indices.keys())[list(train_gen.class_indices.values()).index(class_id)]
    print(f"{class_name}: {count}")
```

    Training Set Class Breakdown:
    Overall: 4571
    notumor: 1276
    glioma: 1057
    meningioma: 1072
    pituitary: 1166
    


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


    
![png](output_55_0.png)
    



```python
##################################
# Creating subsets of images
# for model validation and
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
# Obtaining the count and class distribution
# for the validation set
##################################
total_images = val_gen.samples
class_counts = Counter(val_gen.classes)
print("Validation Set Class Breakdown:")
print(f"Overall: {total_images}")
for class_id, count in class_counts.items():
    class_name = list(val_gen.class_indices.keys())[list(val_gen.class_indices.values()).index(class_id)]
    print(f"{class_name}: {count}")
```

    Validation Set Class Breakdown:
    Overall: 1141
    notumor: 319
    glioma: 264
    meningioma: 267
    pituitary: 291
    


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


    
![png](output_58_0.png)
    



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
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
    Image        object
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
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Te-glTr_0001</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Te-glTr_0002</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Te-glTr_0003</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
      <td>Glioma</td>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Te-glTr_0004</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Testing\gl...</td>
      <td>Te-glTr</td>
      <td>1</td>
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


    
![png](output_62_0.png)
    



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

```


    
![png](output_64_0.png)
    



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


    
![png](output_75_0.png)
    


## 1.5 Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Distinct patterns were observed between the image categories.
    * Images identified with <span style="color: #FF0000">CLASS: No Tumor</span> had the following characteristics:
        * Higher mean pixel values indicating generally lighter images
        * Multimodal and wider distribution of mean pixel values indicating a higher variation
        * Wider range of image pixel standard deviation indicating a higher variation in contrast
    * Images identified with <span style="color: #FF0000">CLASS: Pituitary</span> had the following characteristics:
        * Lower mean pixel values indicating generally darker images
        * Unimodal and steeper distribution of mean pixel values indicating more stable variation
        * Minimal outliers of image pixel standard deviation indicating subset of images with high contrast
    * Images identified with <span style="color: #FF0000">CLASS: Meningioma</span> had the following characteristics:
        * Lower mean pixel values indicating generally darker images
        * Unimodal and steeper distribution of mean pixel values indicating more stable variation
        * Moderate outliers of image pixel standard deviation indicating subset of images with high contrast
    * Images identified with <span style="color: #FF0000">CLASS: Glioma</span> had the following characteristics:
        * Lower mean pixel values indicating generally darker images
        * Unimodal and steeper distribution of mean pixel values indicating more stable variation
        * Compact range of image pixel standard deviation indicating images with stable and sufficient contrast
          


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


    
![png](output_83_0.png)
    



```python
##################################
# Formulating the maximum distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'Max', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Maximum Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_84_0.png)
    



```python
##################################
# Formulating the minimum distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'Min', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Minimum Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_85_0.png)
    



```python
##################################
# Formulating the standard deviation distribution
# by category of the image pixel values
##################################
sns.displot(data = imageEDA, x = 'StDev', kind="kde", hue = 'Class', height=6, aspect=1.40)
plt.title('Image Pixel Intensity Standard Deviation Distribution by Category', fontsize=14, weight='bold');

```


    
![png](output_86_0.png)
    



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


    
![png](output_87_0.png)
    



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


    
![png](output_88_0.png)
    



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


    
![png](output_89_0.png)
    



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


    
![png](output_90_0.png)
    



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


    
![png](output_91_0.png)
    



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


    
![png](output_92_0.png)
    



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


    
![png](output_93_0.png)
    



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


    
![png](output_94_0.png)
    



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


    
![png](output_95_0.png)
    



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


    
![png](output_96_0.png)
    



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


    
![png](output_97_0.png)
    



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


    
![png](output_98_0.png)
    



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


    
![png](output_99_0.png)
    



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


    
![png](output_100_0.png)
    



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


    
![png](output_101_0.png)
    



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


    
![png](output_102_0.png)
    



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


    
![png](output_103_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors (based on the summary statistics for the image pixel values) to the <span style="color: #FF0000">Class</span> event variable assessed collectively was statistically evaluated using the following hypotheses:
    * **Null**: Collective difference in the means or mean ranks between the No Tumor, Pituitary, Meningioma and Glioma groups is equal to zero  
    * **Alternative**: Collective difference in the means or mean ranks between the No Tumor, Pituitary, Meningioma and Glioma groups is not equal to zero    
2. There is sufficient evidence to conclude of a statistically significant difference between the mean ranks of the numeric measurements (based on the summary statistics for the image pixel values) obtained from the <span style="color: #FF0000">Class</span> groups given their high kruskall-wallis statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Mean</span>: Group.Comparison.Test.Statistic=2251.886, Group.Comparison.Test.PValue=0.000
    * <span style="color: #FF0000">St_Dev</span>: Group.Comparison.Test.Statistic=2329.458, Group.Comparison.Test.PValue=0.000 
    * <span style="color: #FF0000">Max</span>: Group.Comparison.Test.Statistic=1982.847, Group.Comparison.Test.PValue=0.000  
    * <span style="color: #FF0000">Min</span>: Group.Comparison.Test.Statistic=394.328, Group.Comparison.Test.PValue=0.000 
3. The relationship between the numeric predictors (based on the summary statistics for the image pixel values) to the <span style="color: #FF0000">Class</span> event variable assessed pairwise was statistically evaluated using the following hypotheses:
    * **Null**: Pairwise difference in the means or mean ranks between the No Tumor, Pituitary, Meningioma and Glioma groups is equal to zero  
    * **Alternative**: Pairwise difference  in the means or mean ranks between the No Tumor, Pituitary, Meningioma and Glioma groups is not equal to zero   
4. There is sufficient evidence to conclude of a statistically significant difference between the mean ranks for <span style="color: #FF0000">Mean</span> as evaluated using all the pairwise combinations of the <span style="color: #FF0000">Class</span> groups given their high mann-whitney U test values with reported low p-values less than the significance level of 0.05.
5. There is sufficient evidence to conclude of a statistically significant difference between the mean ranks for <span style="color: #FF0000">StDev</span> as evaluated using all the pairwise combinations of the <span style="color: #FF0000">Class</span> groups given their high mann-whitney U test values with reported low p-values less than the significance level of 0.05.
6. There is sufficient evidence to conclude of a statistically significant difference between the mean ranks for <span style="color: #FF0000">Max</span> as evaluated using all the pairwise combinations of the <span style="color: #FF0000">Class</span> groups given their high mann-whitney U test values with reported low p-values less than the significance level of 0.05.
7. There is sufficient evidence to conclude of a statistically significant difference between the mean ranks for <span style="color: #FF0000">Min</span> as evaluated using all the pairwise combinations of the <span style="color: #FF0000">Class</span> groups given their high mann-whitney U test values with reported low p-values less than the significance level of 0.05.



```python
##################################
# Displaying the summary statistics
# for the image pixel values
# for hypothesis testing
##################################
imageEDA.head()
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
      <th>Image</th>
      <th>Class</th>
      <th>Path</th>
      <th>Mean</th>
      <th>StDev</th>
      <th>Max</th>
      <th>Min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
      <td>Glioma</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>31.392000</td>
      <td>43.092624</td>
      <td>246</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
      <td>Glioma</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>37.849950</td>
      <td>43.262592</td>
      <td>240</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
      <td>Glioma</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>36.042375</td>
      <td>40.557254</td>
      <td>223</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
      <td>Glioma</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>24.911150</td>
      <td>27.533453</td>
      <td>229</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>
      <td>Glioma</td>
      <td>..\datasets\Brain_Tumor_MRI_Dataset\Training\g...</td>
      <td>32.090825</td>
      <td>33.166552</td>
      <td>236</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the predictor columns
# for statistical evaluation
##################################
predictor_cols = ['Mean', 'StDev', 'Max', 'Min']
statistical_test_results = []
```


```python
##################################
# Checking data assumptions
# and implementing the most appropriate
# group comparison statistical tests
##################################
for col in predictor_cols:
    # Grouping imageEDA by Class
    groups = [group[col].values for name, group in imageEDA.groupby('Class')]
    
    # Performing normality tests using Shapiro-Wilk
    normality = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) > 3)  
    
    # Performing homogeneity of variance test using Levene's test
    homogeneity = stats.levene(*groups)[1] > 0.05 if len(groups) > 1 else True
    
    # Choosing the most appropriate test based on assumptions
    if normality and homogeneity:
        # One-way ANOVA for data meeting the normality and homogeneity assumptions
        test_stat, p_value = stats.f_oneway(*groups)  
        test_name = "ANOVA"
    else:
        # Kruskal-Wallis H-test for data not meeting the normality and homogeneity assumptions
        test_stat, p_value = stats.kruskal(*groups)  
        test_name = "Kruskal-Wallis"
    
    # Appending the statistical_test_results
    statistical_test_results.append({
        'Group.Comparison': f"Class_{col}",
        'Group.Comparison.Test.Statistic': test_stat,
        'Group.Comparison.Test.PValue': p_value,
        'Group.Comparison.Statistical.Test': test_name
    })

```


```python
##################################
# Formulating the group comparison test summary
# between the target variable
# and numeric predictor columns
# comprised of the summary statistics
# for the image pixel values
##################################
statistical_test_summary = pd.DataFrame(statistical_test_results)
statistical_test_summary.set_index('Group.Comparison', inplace=True)
display(statistical_test_summary)
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
      <th>Group.Comparison.Test.Statistic</th>
      <th>Group.Comparison.Test.PValue</th>
      <th>Group.Comparison.Statistical.Test</th>
    </tr>
    <tr>
      <th>Group.Comparison</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Class_Mean</th>
      <td>2251.886072</td>
      <td>0.000000e+00</td>
      <td>Kruskal-Wallis</td>
    </tr>
    <tr>
      <th>Class_StDev</th>
      <td>2329.458509</td>
      <td>0.000000e+00</td>
      <td>Kruskal-Wallis</td>
    </tr>
    <tr>
      <th>Class_Max</th>
      <td>1982.847155</td>
      <td>0.000000e+00</td>
      <td>Kruskal-Wallis</td>
    </tr>
    <tr>
      <th>Class_Min</th>
      <td>394.328915</td>
      <td>3.745680e-85</td>
      <td>Kruskal-Wallis</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Creating a function to conduct
# a pairwise post-hoc evaluation
# of the numeric predictors from the
# group comparison statistical test results
##################################
post_hoc_statistical_test_results = {}

def pairwise_comparisons(imageEDA, col, groups, test_type="ANOVA"):
    pairs = list(itertools.combinations(groups, 2))
    pairwise_statistical_test_results = []
    
    for g1, g2 in pairs:
        group1 = imageEDA[imageEDA['Class'] == g1][col].values
        group2 = imageEDA[imageEDA['Class'] == g2][col].values
        
        if test_type == "ANOVA":
            # Using pairwise comparisons with the Tukey's HSD test for ANOVA results
            test_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        else:
            # Using pairwise comparisons with the Mann-Whitney U test for Kruskal-Wallis H-test results
            test_stat, p_value = stats.mannwhitneyu(group1, group2)
        
        pairwise_statistical_test_results.append({'Group.1': g1,
                                                  'Group.2': g2,
                                                  'PostHoc.Group.Comparison.Test.Statistic': test_stat,
                                                  'PostHoc.Group.Comparison.Test.PValue': p_value})
    
    # Applying Bonferroni correction for the p-values obtained during the pairwise comparisons
    p_values = [r['PostHoc.Group.Comparison.Test.PValue'] for r in pairwise_statistical_test_results]
    corrected = multipletests(p_values, method='bonferroni')
    for r, corr_p in zip(pairwise_statistical_test_results, corrected[1]):
        r['PostHoc.Group.Comparison.Test.Bonferroni.Corrected.PValue'] = corr_p
    
    return pd.DataFrame(pairwise_statistical_test_results)
    
```


```python
##################################
# Implementing the pairwise post-hoc evaluation
# of the numeric predictors from the
# group comparison statistical test results
##################################
for result in statistical_test_results:
    col = result['Group.Comparison'].split('_')[1]
    if result['Group.Comparison.Test.PValue'] < 0.05:
        groups = imageEDA['Class'].unique()
        test_type = "ANOVA" if result['Group.Comparison.Statistical.Test'] == "ANOVA" else "Kruskal-Wallis"
        post_hoc_statistical_test_results[col] = pairwise_comparisons(imageEDA, col, groups, test_type)
```


```python
##################################
# Formulating the summary for 
# the pairwise post-hoc evaluation
# of the numeric predictors from the
# group comparison statistical test results
##################################
for col, df in post_hoc_statistical_test_results.items():
    print(f"Post-Hoc Statistical Test Results for {col}:")
    display(df)
    print("\n")

```

    Post-Hoc Statistical Test Results for Mean:
    


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
      <th>Group.1</th>
      <th>Group.2</th>
      <th>PostHoc.Group.Comparison.Test.Statistic</th>
      <th>PostHoc.Group.Comparison.Test.PValue</th>
      <th>PostHoc.Group.Comparison.Test.Bonferroni.Corrected.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glioma</td>
      <td>Meningioma</td>
      <td>413058.5</td>
      <td>3.368236e-125</td>
      <td>2.020941e-124</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glioma</td>
      <td>No Tumor</td>
      <td>206883.0</td>
      <td>2.789625e-306</td>
      <td>1.673775e-305</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Glioma</td>
      <td>Pituitary</td>
      <td>169398.0</td>
      <td>1.014580e-308</td>
      <td>6.087478e-308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meningioma</td>
      <td>No Tumor</td>
      <td>503954.0</td>
      <td>2.101071e-134</td>
      <td>1.260643e-133</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meningioma</td>
      <td>Pituitary</td>
      <td>525848.5</td>
      <td>1.104420e-98</td>
      <td>6.626523e-98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No Tumor</td>
      <td>Pituitary</td>
      <td>1551655.0</td>
      <td>8.318077e-58</td>
      <td>4.990846e-57</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Post-Hoc Statistical Test Results for StDev:
    


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
      <th>Group.1</th>
      <th>Group.2</th>
      <th>PostHoc.Group.Comparison.Test.Statistic</th>
      <th>PostHoc.Group.Comparison.Test.PValue</th>
      <th>PostHoc.Group.Comparison.Test.Bonferroni.Corrected.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glioma</td>
      <td>Meningioma</td>
      <td>298990.0</td>
      <td>4.995792e-192</td>
      <td>2.997475e-191</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glioma</td>
      <td>No Tumor</td>
      <td>202390.0</td>
      <td>1.619059e-309</td>
      <td>9.714352e-309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Glioma</td>
      <td>Pituitary</td>
      <td>666271.0</td>
      <td>1.112726e-44</td>
      <td>6.676358e-44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meningioma</td>
      <td>No Tumor</td>
      <td>516617.0</td>
      <td>1.593771e-128</td>
      <td>9.562626e-128</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meningioma</td>
      <td>Pituitary</td>
      <td>1439681.0</td>
      <td>4.521296e-105</td>
      <td>2.712778e-104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No Tumor</td>
      <td>Pituitary</td>
      <td>2008320.0</td>
      <td>1.870740e-265</td>
      <td>1.122444e-264</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Post-Hoc Statistical Test Results for Max:
    


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
      <th>Group.1</th>
      <th>Group.2</th>
      <th>PostHoc.Group.Comparison.Test.Statistic</th>
      <th>PostHoc.Group.Comparison.Test.PValue</th>
      <th>PostHoc.Group.Comparison.Test.Bonferroni.Corrected.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glioma</td>
      <td>Meningioma</td>
      <td>478405.0</td>
      <td>1.745474e-93</td>
      <td>1.047284e-92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glioma</td>
      <td>No Tumor</td>
      <td>246714.5</td>
      <td>1.816804e-298</td>
      <td>1.090083e-297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Glioma</td>
      <td>Pituitary</td>
      <td>688685.5</td>
      <td>1.857985e-38</td>
      <td>1.114791e-37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meningioma</td>
      <td>No Tumor</td>
      <td>402750.5</td>
      <td>1.362945e-203</td>
      <td>8.177673e-203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meningioma</td>
      <td>Pituitary</td>
      <td>1154535.0</td>
      <td>4.358408e-17</td>
      <td>2.615045e-16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No Tumor</td>
      <td>Pituitary</td>
      <td>1979668.5</td>
      <td>1.892725e-264</td>
      <td>1.135635e-263</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Post-Hoc Statistical Test Results for Min:
    


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
      <th>Group.1</th>
      <th>Group.2</th>
      <th>PostHoc.Group.Comparison.Test.Statistic</th>
      <th>PostHoc.Group.Comparison.Test.PValue</th>
      <th>PostHoc.Group.Comparison.Test.Bonferroni.Corrected.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glioma</td>
      <td>Meningioma</td>
      <td>869218.0</td>
      <td>1.726123e-06</td>
      <td>1.035674e-05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Glioma</td>
      <td>No Tumor</td>
      <td>933286.5</td>
      <td>8.827701e-37</td>
      <td>5.296621e-36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Glioma</td>
      <td>Pituitary</td>
      <td>961688.0</td>
      <td>3.413685e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Meningioma</td>
      <td>No Tumor</td>
      <td>964530.0</td>
      <td>1.462778e-24</td>
      <td>8.776671e-24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Meningioma</td>
      <td>Pituitary</td>
      <td>991543.0</td>
      <td>2.362945e-06</td>
      <td>1.417767e-05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No Tumor</td>
      <td>Pituitary</td>
      <td>1293690.5</td>
      <td>1.368021e-39</td>
      <td>8.208125e-39</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    

## 1.6 Predictive Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Pre-Modelling Data Preparation <a class="anchor" id="1.6.1"></a>

1. Training data (obtained as a subset of the initial training set representing 80%) included **4571 augmented images**.
   * <span style="color: #FF0000">CLASS: No Tumor</span> = **1276 images**
   * <span style="color: #FF0000">CLASS: Pituitary</span> = **1116 images**
   * <span style="color: #FF0000">CLASS: Meningioma</span> = **1072 images**
   * <span style="color: #FF0000">CLASS: Glioma</span> = **1057 images**
2. Validation data (obtained as a subset of the initial training data representing 20%) included **1141 original images**.
   * <span style="color: #FF0000">CLASS: No Tumor</span> = **319 images**
   * <span style="color: #FF0000">CLASS: Pituitary</span> = **291 images**
   * <span style="color: #FF0000">CLASS: Meningioma</span> = **267 images**
   * <span style="color: #FF0000">CLASS: Glioma</span> = **264 images**
3. Candidate models were formulated using common layers as follows:
    * **Convolutional Layer** (<span style="color: #FF0000">Conv_2D</span>) - extracts features from input images using convolutional filters
    * **Maximum Pooling Layer** (<span style="color: #FF0000">MaxPooling2D</span>) - Reduces spatial dimensions and downsamples feature maps
    * **Activation Layer** (<span style="color: #FF0000">Activation</span>)- Applies an activation function element-wise to the output
    * **Flatten Layer** (<span style="color: #FF0000">Flatten</span>) - Flattens the input to a 1D array, preparing for fully connected layers
    * **Dense Layer** (<span style="color: #FF0000">Dense</span>) - Fully connected layer for classification
4. Different iterations of the model were formulated using variations in the inclusion or exclusion of the following regularization layers:
    * **Dropout Layer** (<span style="color: #FF0000">Droptout</span>) - randomly drops (sets to zero) a fraction of the neurons during training reducing co-dependencies between them
    * **Batch Normalization Layer** (<span style="color: #FF0000">BatchNormalization</span>) - adjusts and scales the inputs to a layer reducing the sensitivity to weight initialization choices
5. A subset of hyperparameters for the different layers were fixed during model training including:
    * <span style="color: #FF0000">kernel_size</span> - setting used to define the local region the convolutional layer considers when processing the input
    * <span style="color: #FF0000">activation</span> - setting used to introduce non-linearity into the model, enabling it to learn complex relationships in the data
    * <span style="color: #FF0000">pool_size</span> - setting used to reduce the spatial dimensions of the feature maps to focus on the most important features
    * <span style="color: #FF0000">padding</span> - setting used to control the spatial size and shape for every convolutional operation at each stage
    * <span style="color: #FF0000">optimizer</span> - setting used to determine how the model's weights are updated during training
    * <span style="color: #FF0000">batch_size</span> - setting used to determine how many samples are used in each iteration of training
    * <span style="color: #FF0000">loss</span> - setting used to define the objective that the model seeks to minimize during training
6. A subset of hyperparameters for the different layers were optimized during model training including:
    * <span style="color: #FF0000">filters</span> - setting used to capture spatial hierarchies and features in the input images
    * <span style="color: #FF0000">units</span> - setting used to process the flattened feature maps and determine the dimensionality of the output space
    * <span style="color: #FF0000">learning_rate</span> - setting used to determine the step size at each iteration during optimization
7. Two CNN model structures were additionally evaluated as follows:
    * **Simple**
        * Lesser number of <span style="color: #FF0000">Conv_2D</span>
        * Lower values set for <span style="color: #FF0000">filters</span>
        * Lower values set for <span style="color: #FF0000">units</span>
    * **Complex**
        * Higher number of <span style="color: #FF0000">Conv_2D</span>
        * Higher values set for <span style="color: #FF0000">filters</span>
        * Higher values set for <span style="color: #FF0000">units</span>


### 1.6.2 Convolutional Neural Network Sequential Layer Development <a class="anchor" id="1.6.2"></a>

1. The [convolutional neural network model](https://www.tensorflow.org/api_docs/python/tf/keras/models) from the <mark style="background-color: #CCECFF"><b>keras.models</b></mark> Python library API was implemented using a [sequential model](https://keras.io/guides/sequential_model/) structure.
2. Specialized functions were applied to fine-tune the training process dynamically and automate certain actions including:
    * <span style="color: #FF0000">EarlyStopping</span> to stop training when a monitored metric stops improving which helps to prevent overfitting and saves time, with fixed hyperparameters as follows:
        * <span style="color: #FF0000">monitor</span> = validation loss (metric to track)
        * <span style="color: #FF0000">patience</span> = 10 (number of epochs with no improvement after which training stops)
        * <span style="color: #FF0000">restore_best_weights</span> = true (model's weights will be restored to those from the epoch with the best value of the monitored metric)
        * <span style="color: #FF0000">min_delta</span> = 0.0001 (minimum change in the monitored metric to qualify as an improvement)
    * <span style="color: #FF0000">ReduceLROnPlateau</span> to reduce the learning rate when a monitored metric has stopped improving which helps the model converge better by fine-tuning updates, with fixed hyperparameters as follows:
        * <span style="color: #FF0000">monitor</span> = validation loss (metric to track)
        * <span style="color: #FF0000">factor</span> = 0.10 (percentage by which the learning rate is reduced)
        * <span style="color: #FF0000">patience</span> = 3 (number of epochs with no improvement after which training stops)
        * <span style="color: #FF0000">min_lr</span> = 0.000001 (lower bound for the learning rate to prevent it from becoming too small)
    * <span style="color: #FF0000">ModelCheckpoint</span> to save the model at specified intervals during training and enabling the subsequent restoration of the best-performing model, with fixed hyperparameters as follows:
        * <span style="color: #FF0000">monitor</span> = validation loss (metric to track)
        * <span style="color: #FF0000">save_best_only</span> = true (only saves the model when the monitored metric improves)
        * <span style="color: #FF0000">save_weights_only</span> = false (saving the entire model and not just the weights)
          


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
    patience=10,                              # Defining the number of epochs to wait before stopping if no improvement
    min_delta=1e-4,                          # Defining the minimum change in the monitored quantity to qualify as an improvement
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

1. The simple model contains 7 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: nr_simple_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 8
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: nr_simple_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: nr_simple_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: nr_simple_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: nr_simple_flatten</span>
    * <span style="color: #FF0000">Dense: nr_simple_dense_0</span>
        * <span style="color: #FF0000">units</span> = 32
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dense: nr_simple_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
2. The complex model contains 9 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 32
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_2</span>
        * <span style="color: #FF0000">filters</span> = 64
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_2</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: nr_complex_flatten</span>
    * <span style="color: #FF0000">Dense: nr_complex_dense_0</span>
        * <span style="color: #FF0000">units</span> = 128
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dense: nr_complex_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
3. Additional fixed hyperparameters used during model compilation are as follows:
    * <span style="color: #FF0000">loss</span> = categorical_crossentropy
    * <span style="color: #FF0000">optimizer</span> = adam (adaptive moment estimation)
    * <span style="color: #FF0000">metrics</span> = recall



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

1. The simple model contains 8 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: dr_simple_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 8
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: dr_simple_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: dr_simple_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: dr_simple_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: dr_simple_flatten</span>
    * <span style="color: #FF0000">Dense: dr_simple_dense_0</span>
        * <span style="color: #FF0000">units</span> = 32
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dropout: dr_simple_dropout</span>
        * <span style="color: #FF0000">rate</span> = 0.25
    * <span style="color: #FF0000">Dense: dr_simple_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
2. The complex model contains 10 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 32
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_2</span>
        * <span style="color: #FF0000">filters</span> = 64
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_2</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: dr_complex_flatten</span>
    * <span style="color: #FF0000">Dense: dr_complex_dense_0</span>
        * <span style="color: #FF0000">units</span> = 128
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dropout: dr_complex_dropout</span>
        * <span style="color: #FF0000">rate</span> = 0.25
    * <span style="color: #FF0000">Dense: dr_complex_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
3. Additional fixed hyperparameters used during model compilation are as follows:
    * <span style="color: #FF0000">loss</span> = categorical_crossentropy
    * <span style="color: #FF0000">optimizer</span> = adam (adaptive moment estimation)
    * <span style="color: #FF0000">metrics</span> = recall



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

1. The simple model contains 9 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: bnr_simple_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 8
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: bnr_simple_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: bnr_simple_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">BatchNormalization: bnr_simple_batch_normalization</span>
    * <span style="color: #FF0000">Activation: bnr_simple_activation</span>
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">MaxPooling2D: bnr_simple_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: bnr_simple_flatten</span>
    * <span style="color: #FF0000">Dense: bnr_simple_dense_0</span>
        * <span style="color: #FF0000">units</span> = 32
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dense: bnr_simple_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
2. The complex model contains 10 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 32
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">BatchNormalization: bnr_complex_batch_normalization</span>
    * <span style="color: #FF0000">Activation: bnr_complex_activation</span>
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_2</span>
        * <span style="color: #FF0000">filters</span> = 64
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_2</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: bnr_complex_flatten</span>
    * <span style="color: #FF0000">Dense: bnr_complex_dense_0</span>
        * <span style="color: #FF0000">units</span> = 128
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dense: bnr_complex_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
3. Additional fixed hyperparameters used during model compilation are as follows:
    * <span style="color: #FF0000">loss</span> = categorical_crossentropy
    * <span style="color: #FF0000">optimizer</span> = adam (adaptive moment estimation)
    * <span style="color: #FF0000">metrics</span> = recall



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

1. The simple model contains 10 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: cdrbnr_simple_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 8
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_simple_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: cdrbnr_simple_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">BatchNormalization: cdrbnr_simple_batch_normalization</span>
    * <span style="color: #FF0000">Activation: cdrbnr_simple_activation</span>
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_simple_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: cdrbnr_simple_flatten</span>
    * <span style="color: #FF0000">Dense: cdrbnr_simple_dense_0</span>
        * <span style="color: #FF0000">units</span> = 32
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dropout: cdrbnr_simple_dropout</span>
        * <span style="color: #FF0000">rate</span> = 0.25
    * <span style="color: #FF0000">Dense: cdrbnr_simple_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
2. The complex model contains 11 layers with fixed hyperparameters as follows:
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_0</span>
        * <span style="color: #FF0000">filters</span> = 16
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
        * <span style="color: #FF0000">input_shape</span> = 227x227x1
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_0</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_1</span>
        * <span style="color: #FF0000">filters</span> = 32
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">BatchNormalization: cdrbnr_complex_batch_normalization</span>
    * <span style="color: #FF0000">Activation: cdrbnr_complex_activation</span>
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_1</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_2</span>
        * <span style="color: #FF0000">filters</span> = 64
        * <span style="color: #FF0000">kernel_size</span> = 3x3
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
        * <span style="color: #FF0000">padding</span> = same (output size equals input size)
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_2</span>
        * <span style="color: #FF0000">pool_size</span> = 2x2
    * <span style="color: #FF0000">Flatten: cdrbnr_complex_flatten</span>
    * <span style="color: #FF0000">Dense: cdrbnr_complex_dense_0</span>
        * <span style="color: #FF0000">units</span> = 128
        * <span style="color: #FF0000">activation</span> = relu (rectified linear unit)
    * <span style="color: #FF0000">Dropout: cdrbnr_complex_dropout</span>
        * <span style="color: #FF0000">rate</span> = 0.25
    * <span style="color: #FF0000">Dense: cdrbnr_complex_dense_1</span>
        * <span style="color: #FF0000">units</span> = 4
        * <span style="color: #FF0000">activation</span> = softmax
3. Additional fixed hyperparameters used during model compilation are as follows:
    * <span style="color: #FF0000">loss</span> = categorical_crossentropy
    * <span style="color: #FF0000">optimizer</span> = adam (adaptive moment estimation)
    * <span style="color: #FF0000">metrics</span> = recall



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

1. The simple model contained 1,607,044 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: nr_simple_conv2d_0</span>
        * output size = 227x227x8
        * number of parameters = 80
    * <span style="color: #FF0000">MaxPooling2D: nr_simple_max_pooling2d_0</span>
        * output size = 113x113x8
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: nr_simple_conv2d_1</span>
        * output size = 113x113x16
        * number of parameters = 1,168 
    * <span style="color: #FF0000">MaxPooling2D: nr_simple_max_pooling2d_1</span>
        * output size = 56x56x16
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: nr_simple_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: nr_simple_dense_0</span>
        * output size = 32
        * number of parameters = 1,605,664
    * <span style="color: #FF0000">Dense: nr_simple_dense_1</span>
        * output size = 4
        * number of parameters = 132
2. The complex model contained 6,446,468 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_0</span>
        * output size = 227x227x16
        * number of parameters = 160
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_0</span>
        * output size = 113x113x16
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_1</span>
        * output size = 113x113x32
        * number of parameters = 4,640
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_1</span>
        * output size = 56x56x32
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: nr_complex_conv2d_2</span>
        * output size = 56x56x64
        * number of parameters = 18,496 
    * <span style="color: #FF0000">MaxPooling2D: nr_complex_max_pooling2d_2</span>
        * output size = 28x28x64
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: nr_complex_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: nr_complex_dense_0</span>
        * output size = 128
        * number of parameters = 6,422,656
    * <span style="color: #FF0000">Dense: nr_complex_dense_1</span>
        * output size = 4
        * number of parameters = 516
3. The model performance on the validation set for all image categories is summarized as follows:
    * Simple
        * **Precision** = 0.8047
        * **Recall** = 0.7926
        * **F1 Score** = 0.7946
    * Complex
        * **Precision** = 0.7963
        * **Recall** = 0.7960
        * **F1 Score** = 0.7944



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
model_nr_simple.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 261ms/step - loss: 0.8697 - recall: 0.4612 - val_loss: 0.9379 - val_recall: 0.6556 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 252ms/step - loss: 0.4279 - recall: 0.8129 - val_loss: 0.8684 - val_recall: 0.6792 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 254ms/step - loss: 0.3339 - recall: 0.8637 - val_loss: 0.8071 - val_recall: 0.7239 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 246ms/step - loss: 0.3062 - recall: 0.8771 - val_loss: 0.9367 - val_recall: 0.7528 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 249ms/step - loss: 0.2505 - recall: 0.9024 - val_loss: 0.8099 - val_recall: 0.7450 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m67s[0m 466ms/step - loss: 0.2282 - recall: 0.9033 - val_loss: 0.7319 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.1857 - recall: 0.9301 - val_loss: 0.8285 - val_recall: 0.7783 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 254ms/step - loss: 0.1783 - recall: 0.9361 - val_loss: 0.8437 - val_recall: 0.7642 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 255ms/step - loss: 0.1366 - recall: 0.9491 - val_loss: 0.8675 - val_recall: 0.8089 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 259ms/step - loss: 0.1127 - recall: 0.9611 - val_loss: 0.7600 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 254ms/step - loss: 0.0880 - recall: 0.9663 - val_loss: 0.7769 - val_recall: 0.8177 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m132s[0m 921ms/step - loss: 0.1055 - recall: 0.9612 - val_loss: 0.7722 - val_recall: 0.8221 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.0787 - recall: 0.9733 - val_loss: 0.7732 - val_recall: 0.8221 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 264ms/step - loss: 0.0926 - recall: 0.9680 - val_loss: 0.7768 - val_recall: 0.8221 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.0824 - recall: 0.9691 - val_loss: 0.7803 - val_recall: 0.8212 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 249ms/step - loss: 0.0939 - recall: 0.9701 - val_loss: 0.7808 - val_recall: 0.8203 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with no regularization
# on the independent validation set
##################################
model_nr_simple_y_pred_val = model_nr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 112ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with no regularization
# on the training and validation sets
##################################
plot_training_history(model_nr_simple_history, 'Simple CNN With No Regularization : ')

```


    
![png](output_166_0.png)
    



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
    
    


    
![png](output_167_1.png)
    



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
model_nr_complex.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m59s[0m 399ms/step - loss: 1.0913 - recall: 0.3645 - val_loss: 0.8411 - val_recall: 0.6915 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 373ms/step - loss: 0.4091 - recall: 0.8322 - val_loss: 0.8689 - val_recall: 0.6862 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 374ms/step - loss: 0.2674 - recall: 0.8948 - val_loss: 0.8096 - val_recall: 0.7327 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 389ms/step - loss: 0.2156 - recall: 0.9202 - val_loss: 0.8086 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 372ms/step - loss: 0.1748 - recall: 0.9339 - val_loss: 0.8040 - val_recall: 0.7625 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.1469 - recall: 0.9431 - val_loss: 0.7236 - val_recall: 0.7984 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 375ms/step - loss: 0.1025 - recall: 0.9621 - val_loss: 0.7801 - val_recall: 0.7993 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 376ms/step - loss: 0.0918 - recall: 0.9644 - val_loss: 0.9317 - val_recall: 0.8063 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m52s[0m 364ms/step - loss: 0.0861 - recall: 0.9650 - val_loss: 0.8448 - val_recall: 0.8238 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 369ms/step - loss: 0.0560 - recall: 0.9774 - val_loss: 0.8052 - val_recall: 0.8300 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 373ms/step - loss: 0.0285 - recall: 0.9933 - val_loss: 0.8621 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m100s[0m 697ms/step - loss: 0.0294 - recall: 0.9919 - val_loss: 0.8798 - val_recall: 0.8256 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 386ms/step - loss: 0.0235 - recall: 0.9925 - val_loss: 0.8846 - val_recall: 0.8230 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 377ms/step - loss: 0.0297 - recall: 0.9903 - val_loss: 0.8888 - val_recall: 0.8247 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 378ms/step - loss: 0.0237 - recall: 0.9951 - val_loss: 0.9018 - val_recall: 0.8230 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 379ms/step - loss: 0.0283 - recall: 0.9913 - val_loss: 0.9021 - val_recall: 0.8238 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with no regularization
# on the independent validation set
##################################
model_nr_complex_y_pred_val = model_nr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 147ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with no regularization
# on the training and validation sets
##################################
plot_training_history(model_nr_complex_history, 'Complex CNN With No Regularization : ')

```


    
![png](output_173_0.png)
    



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


    
![png](output_174_0.png)
    



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

1. The simple model contained 1,607,044 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: dr_simple_conv2d_0</span>
        * output size = 227x227x8
        * number of parameters = 80
    * <span style="color: #FF0000">MaxPooling2D: dr_simple_max_pooling2d_0</span>
        * output size = 113x113x8
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: dr_simple_conv2d_1</span>
        * output size = 113x113x16
        * number of parameters = 1,168 
    * <span style="color: #FF0000">MaxPooling2D: dr_simple_max_pooling2d_1</span>
        * output size = 56x56x16
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: dr_simple_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: dr_simple_dense_0</span>
        * output size = 32
        * number of parameters = 1,605,664
    * <span style="color: #FF0000">Dropout: dr_simple_dropout</span>
        * output size = 32
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: dr_simple_dense_1</span>
        * output size = 4
        * number of parameters = 132
2. The complex model contained 6,446,468 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_0</span>
        * output size = 227x227x16
        * number of parameters = 160
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_0</span>
        * output size = 113x113x16
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_1</span>
        * output size = 113x113x32
        * number of parameters = 4,640
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_1</span>
        * output size = 56x56x32
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: dr_complex_conv2d_2</span>
        * output size = 56x56x64
        * number of parameters = 18,496 
    * <span style="color: #FF0000">MaxPooling2D: dr_complex_max_pooling2d_2</span>
        * output size = 28x28x64
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: dr_complex_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: dr_complex_dense_0</span>
        * output size = 128
        * number of parameters = 6,422,656
    * <span style="color: #FF0000">Dropout: dr_complex_dropout</span>
        * output size = 128
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: dr_complex_dense_1</span>
        * output size = 4
        * number of parameters = 516
3. The model performance on the validation set for all image categories is summarized as follows:
    * Simple
        * **Precision** = 0.7709
        * **Recall** = 0.7576
        * **F1 Score** = 0.7611
    * Complex
        * **Precision** = 0.7878
        * **Recall** = 0.7897
        * **F1 Score** = 0.7881




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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 249ms/step - loss: 1.3558 - recall: 0.1436 - val_loss: 1.0029 - val_recall: 0.4259 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.7573 - recall: 0.5541 - val_loss: 0.8809 - val_recall: 0.5995 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 259ms/step - loss: 0.6801 - recall: 0.5991 - val_loss: 0.8098 - val_recall: 0.6784 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 246ms/step - loss: 0.5949 - recall: 0.6555 - val_loss: 0.9510 - val_recall: 0.6319 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 252ms/step - loss: 0.5358 - recall: 0.6888 - val_loss: 0.8406 - val_recall: 0.6687 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 247ms/step - loss: 0.5175 - recall: 0.7039 - val_loss: 0.7385 - val_recall: 0.6950 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 253ms/step - loss: 0.5096 - recall: 0.7264 - val_loss: 0.8432 - val_recall: 0.7108 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m46s[0m 319ms/step - loss: 0.5263 - recall: 0.7275 - val_loss: 0.7060 - val_recall: 0.7432 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 243ms/step - loss: 0.4338 - recall: 0.7747 - val_loss: 0.8316 - val_recall: 0.7546 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 246ms/step - loss: 0.4617 - recall: 0.7647 - val_loss: 0.8108 - val_recall: 0.7432 - learning_rate: 0.0010
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 254ms/step - loss: 0.4197 - recall: 0.7834 - val_loss: 0.8501 - val_recall: 0.7406 - learning_rate: 0.0010
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 258ms/step - loss: 0.4121 - recall: 0.7925 - val_loss: 0.7721 - val_recall: 0.7634 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 248ms/step - loss: 0.3817 - recall: 0.8064 - val_loss: 0.7482 - val_recall: 0.7713 - learning_rate: 1.0000e-04
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m111s[0m 773ms/step - loss: 0.3763 - recall: 0.8102 - val_loss: 0.7683 - val_recall: 0.7634 - learning_rate: 1.0000e-04
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 248ms/step - loss: 0.3781 - recall: 0.7994 - val_loss: 0.7877 - val_recall: 0.7642 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 253ms/step - loss: 0.3701 - recall: 0.8088 - val_loss: 0.7936 - val_recall: 0.7642 - learning_rate: 1.0000e-05
    Epoch 17/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 251ms/step - loss: 0.3933 - recall: 0.8056 - val_loss: 0.7841 - val_recall: 0.7660 - learning_rate: 1.0000e-05
    Epoch 18/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 250ms/step - loss: 0.3852 - recall: 0.7920 - val_loss: 0.7832 - val_recall: 0.7660 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with dropout regularization
# on the independent validation set
##################################
model_dr_simple_y_pred_val = model_dr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 104ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with dropout regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_simple_history, 'Simple CNN With Dropout Regularization : ')

```


    
![png](output_181_0.png)
    



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


    
![png](output_182_0.png)
    



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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 387ms/step - loss: 1.0131 - recall: 0.3707 - val_loss: 0.8088 - val_recall: 0.6994 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 374ms/step - loss: 0.4345 - recall: 0.8110 - val_loss: 0.7967 - val_recall: 0.6968 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 377ms/step - loss: 0.2910 - recall: 0.8898 - val_loss: 0.7494 - val_recall: 0.7458 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 378ms/step - loss: 0.2426 - recall: 0.9008 - val_loss: 0.7891 - val_recall: 0.7511 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 399ms/step - loss: 0.1822 - recall: 0.9304 - val_loss: 0.6271 - val_recall: 0.7844 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 387ms/step - loss: 0.1632 - recall: 0.9328 - val_loss: 0.7265 - val_recall: 0.7774 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 385ms/step - loss: 0.1317 - recall: 0.9478 - val_loss: 0.8423 - val_recall: 0.7862 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 388ms/step - loss: 0.1286 - recall: 0.9583 - val_loss: 0.8516 - val_recall: 0.8107 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 384ms/step - loss: 0.0860 - recall: 0.9707 - val_loss: 0.7973 - val_recall: 0.8124 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 381ms/step - loss: 0.0758 - recall: 0.9745 - val_loss: 0.8234 - val_recall: 0.8081 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 378ms/step - loss: 0.0523 - recall: 0.9825 - val_loss: 0.8551 - val_recall: 0.8098 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 391ms/step - loss: 0.0571 - recall: 0.9813 - val_loss: 0.8562 - val_recall: 0.8054 - learning_rate: 1.0000e-05
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 397ms/step - loss: 0.0540 - recall: 0.9823 - val_loss: 0.8620 - val_recall: 0.8089 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.0564 - recall: 0.9793 - val_loss: 0.8652 - val_recall: 0.8098 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 375ms/step - loss: 0.0538 - recall: 0.9812 - val_loss: 0.8655 - val_recall: 0.8098 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with dropout regularization
# on the independent validation set
##################################
model_dr_complex_y_pred_val = model_dr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 133ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with dropout regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_complex_history, 'Complex CNN With Dropout Regularization : ')

```


    
![png](output_188_0.png)
    



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


    
![png](output_189_0.png)
    



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

1. The simple model contained 1,607,076 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: bnr_simple_conv2d_0</span>
        * output size = 227x227x8
        * number of parameters = 80
    * <span style="color: #FF0000">MaxPooling2D: bnr_simple_max_pooling2d_0</span>
        * output size = 113x113x8
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: bnr_simple_conv2d_1</span>
        * output size = 113x113x16
        * number of parameters = 1,168
    * <span style="color: #FF0000">BatchNormalization: bnr_simple_batch_normalization</span>
        * output size = 113x113x16
        * number of parameters = 64
    * <span style="color: #FF0000">Activation: bnr_simple_activation</span>
        * output size = 113x113x16
        * number of parameters = 0 
    * <span style="color: #FF0000">MaxPooling2D: bnr_simple_max_pooling2d_1</span>
        * output size = 56x56x16
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: bnr_simple_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: bnr_simple_dense_0</span>
        * output size = 32
        * number of parameters = 1,605,664
    * <span style="color: #FF0000">Dense: bnr_simple_dense_1</span>
        * output size = 4
        * number of parameters = 132
2. The complex model contained 6,446,596 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_0</span>
        * output size = 227x227x16
        * number of parameters = 160
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_0</span>
        * output size = 113x113x16
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_1</span>
        * output size = 113x113x32
        * number of parameters = 4,640
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_1</span>
        * output size = 56x56x32
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: bnr_complex_conv2d_2</span>
        * output size = 56x56x64
        * number of parameters = 18,496
    * <span style="color: #FF0000">BatchNormalization: bnr_complex_batch_normalization</span>
        * output size = 56x56x64
        * number of parameters = 256
    * <span style="color: #FF0000">Activation: bnr_complex_activation</span>
        * output size = 56x56x64
        * number of parameters = 0 
    * <span style="color: #FF0000">MaxPooling2D: bnr_complex_max_pooling2d_2</span>
        * output size = 28x28x64
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: bnr_complex_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: bnr_complex_dense_0</span>
        * output size = 128
        * number of parameters = 6,422,656
    * <span style="color: #FF0000">Dense: bnr_complex_dense_1</span>
        * output size = 4
        * number of parameters = 516
3. The model performance on the validation set for all image categories is summarized as follows:
    * Simple
        * **Precision** = 0.8326
        * **Recall** = 0.8261
        * **F1 Score** = 0.8285
    * Complex
        * **Precision** = 0.6667
        * **Recall** = 0.6539
        * **F1 Score** = 0.6482



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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 276ms/step - loss: 1.7668 - recall: 0.5558 - val_loss: 1.0888 - val_recall: 0.0473 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 273ms/step - loss: 0.3585 - recall: 0.8676 - val_loss: 0.8608 - val_recall: 0.3716 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m43s[0m 286ms/step - loss: 0.2334 - recall: 0.9148 - val_loss: 0.7054 - val_recall: 0.6591 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 289ms/step - loss: 0.2119 - recall: 0.9237 - val_loss: 0.5743 - val_recall: 0.8089 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 288ms/step - loss: 0.2065 - recall: 0.9280 - val_loss: 0.6802 - val_recall: 0.8072 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 289ms/step - loss: 0.1448 - recall: 0.9461 - val_loss: 0.8415 - val_recall: 0.8387 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 281ms/step - loss: 0.1309 - recall: 0.9561 - val_loss: 1.1974 - val_recall: 0.8107 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 275ms/step - loss: 0.0820 - recall: 0.9706 - val_loss: 0.9800 - val_recall: 0.8282 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 284ms/step - loss: 0.0649 - recall: 0.9801 - val_loss: 1.0222 - val_recall: 0.8309 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 275ms/step - loss: 0.0683 - recall: 0.9782 - val_loss: 1.0025 - val_recall: 0.8247 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 281ms/step - loss: 0.0509 - recall: 0.9825 - val_loss: 0.9991 - val_recall: 0.8309 - learning_rate: 1.0000e-05
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m66s[0m 460ms/step - loss: 0.0648 - recall: 0.9745 - val_loss: 0.9882 - val_recall: 0.8309 - learning_rate: 1.0000e-05
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 273ms/step - loss: 0.0472 - recall: 0.9859 - val_loss: 0.9759 - val_recall: 0.8300 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 276ms/step - loss: 0.0499 - recall: 0.9851 - val_loss: 0.9774 - val_recall: 0.8309 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a simple CNN with batch normalization regularization
# on the independent validation set
##################################
model_bnr_simple_y_pred_val = model_bnr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 109ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_bnr_simple_history, 'Simple CNN With Batch Normalization Regularization : ')

```


    
![png](output_196_0.png)
    



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


    
![png](output_197_0.png)
    



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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 405ms/step - loss: 2.4198 - recall: 0.4782 - val_loss: 1.1481 - val_recall: 0.0096 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 400ms/step - loss: 0.3966 - recall: 0.8304 - val_loss: 0.9454 - val_recall: 0.1613 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m83s[0m 406ms/step - loss: 0.2384 - recall: 0.9055 - val_loss: 0.7357 - val_recall: 0.5819 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m83s[0m 410ms/step - loss: 0.2179 - recall: 0.9136 - val_loss: 0.6788 - val_recall: 0.7809 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m79s[0m 392ms/step - loss: 0.1693 - recall: 0.9332 - val_loss: 0.8541 - val_recall: 0.7064 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 394ms/step - loss: 0.1205 - recall: 0.9529 - val_loss: 0.8922 - val_recall: 0.7774 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 406ms/step - loss: 0.1140 - recall: 0.9631 - val_loss: 1.1084 - val_recall: 0.7695 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 394ms/step - loss: 0.0670 - recall: 0.9783 - val_loss: 0.8778 - val_recall: 0.8151 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m142s[0m 994ms/step - loss: 0.0429 - recall: 0.9854 - val_loss: 0.8952 - val_recall: 0.8186 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 392ms/step - loss: 0.0439 - recall: 0.9852 - val_loss: 0.8729 - val_recall: 0.8335 - learning_rate: 1.0000e-04
    


```python
##################################
# Evaluating the model
# for a complex CNN with batch normalization regularization
# on the independent validation set
##################################
model_bnr_complex_y_pred_val = model_bnr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 128ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_bnr_complex_history, 'Complex CNN With Batch Normalization Regularization : ')

```


    
![png](output_203_0.png)
    



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


    
![png](output_204_0.png)
    



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

1. The simple model contained 1,607,076 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: cdrbnr_simple_conv2d_0</span>
        * output size = 227x227x8
        * number of parameters = 80
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_simple_max_pooling2d_0</span>
        * output size = 113x113x8
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: cdrbnr_simple_conv2d_1</span>
        * output size = 113x113x16
        * number of parameters = 1,168
    * <span style="color: #FF0000">BatchNormalization: cdrbnr_simple_batch_normalization</span>
        * output size = 113x113x16
        * number of parameters = 64
    * <span style="color: #FF0000">Activation: cdrbnr_simple_activation</span>
        * output size = 113x113x16
        * number of parameters = 0 
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_simple_max_pooling2d_1</span>
        * output size = 56x56x16
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: cdrbnr_simple_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: cdrbnr_simple_dense_0</span>
        * output size = 32
        * number of parameters = 1,605,664
    * <span style="color: #FF0000">Dropout: cdrbnr_simple_dropout</span>
        * output size = 32
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: cdrbnr_simple_dense_1</span>
        * output size = 4
        * number of parameters = 132
2. The complex model contained 6,446,596 trainable parameters broken down per layer as follows:
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_0</span>
        * output size = 227x227x16
        * number of parameters = 160
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_0</span>
        * output size = 113x113x16
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_1</span>
        * output size = 113x113x32
        * number of parameters = 4,640
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_1</span>
        * output size = 56x56x32
        * number of parameters = 0
    * <span style="color: #FF0000">Conv2D: cdrbnr_complex_conv2d_2</span>
        * output size = 56x56x64
        * number of parameters = 18,496
    * <span style="color: #FF0000">BatchNormalization: cdrbnr_complex_batch_normalization</span>
        * output size = 56x56x64
        * number of parameters = 256
    * <span style="color: #FF0000">Activation: cdrbnr_complex_activation</span>
        * output size = 56x56x64
        * number of parameters = 0 
    * <span style="color: #FF0000">MaxPooling2D: cdrbnr_complex_max_pooling2d_2</span>
        * output size = 28x28x64
        * number of parameters = 0
    * <span style="color: #FF0000">Flatten: cdrbnr_complex_flatten</span>
        * output size = 50,176
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: cdrbnr_complex_dense_0</span>
        * output size = 128
        * number of parameters = 6,422,656
    * <span style="color: #FF0000">Dropout: cdrbnr_complex_dropout</span>
        * output size = 128
        * number of parameters = 0
    * <span style="color: #FF0000">Dense: cdrbnr_complex_dense_1</span>
        * output size = 4
        * number of parameters = 516
3. The model performance on the validation set for all image categories is summarized as follows:
    * Simple
        * **Precision** = 0.6003
        * **Recall** = 0.4639
        * **F1 Score** = 0.4181
    * Complex
        * **Precision** = 0.8429
        * **Recall** = 0.8385
        * **F1 Score** = 0.8388



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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 280ms/step - loss: 1.6579 - recall: 0.1515 - val_loss: 1.3345 - val_recall: 0.0018 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 290ms/step - loss: 1.0206 - recall: 0.3417 - val_loss: 1.1807 - val_recall: 0.0649 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m83s[0m 296ms/step - loss: 0.9324 - recall: 0.3955 - val_loss: 1.0523 - val_recall: 0.2366 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 283ms/step - loss: 0.7758 - recall: 0.4966 - val_loss: 0.9607 - val_recall: 0.4137 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 282ms/step - loss: 0.7319 - recall: 0.5117 - val_loss: 1.0513 - val_recall: 0.4496 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m60s[0m 418ms/step - loss: 0.6944 - recall: 0.5397 - val_loss: 1.0002 - val_recall: 0.5127 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 285ms/step - loss: 0.6810 - recall: 0.5275 - val_loss: 1.1606 - val_recall: 0.6056 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 276ms/step - loss: 0.6298 - recall: 0.5520 - val_loss: 0.9720 - val_recall: 0.5951 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 284ms/step - loss: 0.5942 - recall: 0.5613 - val_loss: 0.9829 - val_recall: 0.5960 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 283ms/step - loss: 0.6268 - recall: 0.5480 - val_loss: 1.0679 - val_recall: 0.5942 - learning_rate: 1.0000e-04
    


```python
##################################
# Evaluating the model
# for a simple CNN with dropout and batch normalization regularization
# on the independent validation set
##################################
model_cdrbnr_simple_y_pred_val = model_cdrbnr_simple.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 116ms/step
    


```python
##################################
# Plotting the loss profile
# for a simple CNN with dropout and batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_cdrbnr_simple_history, 'Simple CNN With Dropout and Batch Normalization Regularization : ')

```


    
![png](output_211_0.png)
    



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


    
![png](output_212_0.png)
    



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
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m61s[0m 408ms/step - loss: 1.7995 - recall: 0.5219 - val_loss: 1.1321 - val_recall: 0.0342 - learning_rate: 0.0010
    Epoch 2/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m80s[0m 396ms/step - loss: 0.3938 - recall: 0.8333 - val_loss: 0.9887 - val_recall: 0.0649 - learning_rate: 0.0010
    Epoch 3/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 394ms/step - loss: 0.2484 - recall: 0.8988 - val_loss: 0.6290 - val_recall: 0.6713 - learning_rate: 0.0010
    Epoch 4/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 405ms/step - loss: 0.2268 - recall: 0.9093 - val_loss: 0.6252 - val_recall: 0.7555 - learning_rate: 0.0010
    Epoch 5/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 401ms/step - loss: 0.1590 - recall: 0.9359 - val_loss: 0.8430 - val_recall: 0.7046 - learning_rate: 0.0010
    Epoch 6/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m63s[0m 440ms/step - loss: 0.1436 - recall: 0.9409 - val_loss: 0.5680 - val_recall: 0.8352 - learning_rate: 0.0010
    Epoch 7/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 394ms/step - loss: 0.1110 - recall: 0.9563 - val_loss: 0.7335 - val_recall: 0.8344 - learning_rate: 0.0010
    Epoch 8/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 394ms/step - loss: 0.1024 - recall: 0.9607 - val_loss: 0.9613 - val_recall: 0.8291 - learning_rate: 0.0010
    Epoch 9/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 391ms/step - loss: 0.1047 - recall: 0.9615 - val_loss: 0.6784 - val_recall: 0.8475 - learning_rate: 0.0010
    Epoch 10/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 393ms/step - loss: 0.0612 - recall: 0.9779 - val_loss: 0.7055 - val_recall: 0.8580 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 399ms/step - loss: 0.0465 - recall: 0.9825 - val_loss: 0.7504 - val_recall: 0.8615 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m89s[0m 617ms/step - loss: 0.0426 - recall: 0.9825 - val_loss: 0.8035 - val_recall: 0.8624 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 404ms/step - loss: 0.0373 - recall: 0.9885 - val_loss: 0.7971 - val_recall: 0.8624 - learning_rate: 1.0000e-05
    Epoch 14/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 395ms/step - loss: 0.0427 - recall: 0.9842 - val_loss: 0.7896 - val_recall: 0.8606 - learning_rate: 1.0000e-05
    Epoch 15/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 396ms/step - loss: 0.0323 - recall: 0.9903 - val_loss: 0.7911 - val_recall: 0.8606 - learning_rate: 1.0000e-05
    Epoch 16/20
    [1m144/144[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 393ms/step - loss: 0.0418 - recall: 0.9818 - val_loss: 0.7901 - val_recall: 0.8606 - learning_rate: 1.0000e-06
    


```python
##################################
# Evaluating the model
# for a complex CNN with dropout and batch normalization regularization
# on the independent validation set
##################################
model_cdrbnr_complex_y_pred_val = model_cdrbnr_complex.predict(val_gen)

```

    [1m36/36[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 145ms/step
    


```python
##################################
# Plotting the loss profile
# for a complex CNN with dropout and batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_cdrbnr_complex_history, 'Complex CNN With Dropout and Batch Normalization Regularization : ')

```


    
![png](output_218_0.png)
    



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


    
![png](output_219_0.png)
    



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

1. The **Simple CNN Model With No Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.7709
    * **Recall** = 0.7576
    * **F1 Score** = 0.7611
2. The **Complex CNN Model With No Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.7878
    * **Recall** = 0.7897
    * **F1 Score** = 0.7881  
3. The **Simple CNN Model With Dropout Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.7709
    * **Recall** = 0.7576
    * **F1 Score** = 0.7611
4. The **Complex CNN Model With Dropout Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.7878
    * **Recall** = 0.7897
    * **F1 Score** = 0.7881
5. The **Simple CNN Model With Batch Normalization Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.8326
    * **Recall** = 0.8261
    * **F1 Score** = 0.8285
6. The **Complex CNN Model With Batch Normalization Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.6667
    * **Recall** = 0.6539
    * **F1 Score** = 0.6482
7. The **Simple CNN Model With Dropout and Batch Normalization Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.6003
    * **Recall** = 0.4639
    * **F1 Score** = 0.4181
8. The **Complex CNN Model With Dropout and Batch Normalization Regularization** demonstrated the following validation set performance for all image categories:
    * **Precision** = 0.8429
    * **Recall** = 0.8385
    * **F1 Score** = 0.8388
9. The candidate models demonstrating the best validation set performance were as follows:
    * **Simple CNN Model With Batch Normalization Regularization**
        * **Precision** = 0.8326
        * **Recall** = 0.8261
        * **F1 Score** = 0.8285
    * **Complex CNN Model With Dropout and Batch Normalization Regularization** 
        * **Precision** = 0.8429
        * **Recall** = 0.8385
        * **F1 Score** = 0.8388



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


    
![png](output_226_0.png)
    



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


    
![png](output_229_0.png)
    



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


    
![png](output_232_0.png)
    


### 1.6.8 Model Testing <a class="anchor" id="1.6.8"></a>

1. The classification performance of the first selected model - **Simple CNN Model With Batch Normalization Regularization** on the validation set was previously determined as follows:
    * **Precision** = 0.8326
    * **Recall** = 0.8261
    * **F1 Score** = 0.8285
2. The **Simple CNN Model With Batch Normalization Regularization** demonstrated consistent performance on the independent test set as follows:
    * **Precision** = 0.8702
    * **Recall** = 0.8647
    * **F1 Score** = 0.8664
3. The classification performance of the second selected model - **Complex CNN Model With Dropout and Batch Normalization Regularization** on the validation set was previously determined as follows:
    * **Precision** = 0.8429
    * **Recall** = 0.8385
    * **F1 Score** = 0.8388
4. The **Complex CNN Model With Dropout and Batch Normalization Regularization** demonstrated consistent performance on the independent test set as follows:
    * **Precision** = 0.8966
    * **Recall** = 0.8858
    * **F1 Score** = 0.8880
5. The **Complex CNN Model With Dropout and Batch Normalization Regularization** was selected as the final model as it performed better than the competing model.
6. While the classification results have been sufficiently high, the current study can be further extended to achieve optimal model performance through the following:
    * Leverage pre-trained models that have been trained on large datasets to improve performance
    * Conduct further model hyperparameter tuning given sufficient analysis time and higher computing power
    * Formulate deeper neural network architectures to better capture spatial hierarchies and features in the input images
    * Apply other more advanced techniques to interpret the CNN models by understanding and visualizing the features and decisions made at each layer 
    * Consider an imbalanced dataset and apply remedial measures to address unbalanced classification to accurately reflect real-world scenario




```python
##################################
# Evaluating the model
# for a simple CNN with batch normalization regularization
# on the independent test set
##################################
model_bnr_simple_y_pred_test = model_bnr_simple.predict(test_gen)

```

    [1m41/41[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 84ms/step
    


```python
##################################
# Consolidating the predictions
# for a simple CNN with batch normalization regularization
# on the test set
##################################
model_bnr_simple_predictions_test = np.array(list(map(lambda x: np.argmax(x), model_bnr_simple_y_pred_test)))
model_bnr_simple_y_true_test = test_gen.classes

##################################
# Formulating the confusion matrix
# for a simple CNN with batch normalization regularization
# on the test set
##################################
cmatrix_test = pd.DataFrame(confusion_matrix(model_bnr_simple_y_true_test, model_bnr_simple_predictions_test), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for each category of the test set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(cmatrix_test, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('Simple CNN With Batch Normalization Regularization : Test Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

```


    
![png](output_235_0.png)
    



```python
##################################
# Calculating the model accuracy
# for a simple CNN with batch normalization regularization
# for the entire test set
##################################
model_bnr_simple_acc_test = accuracy_score(model_bnr_simple_y_true_test, model_bnr_simple_predictions_test)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for the entire test set
##################################
model_bnr_simple_results_all_test = precision_recall_fscore_support(model_bnr_simple_y_true_test, model_bnr_simple_predictions_test, average='macro', zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for a simple CNN with batch normalization regularization
# for each category of the test set
##################################
model_bnr_simple_results_class_test = precision_recall_fscore_support(model_bnr_simple_y_true_test, model_bnr_simple_predictions_test, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for a simple CNN with batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_bnr_simple_all_df_test = pd.concat([pd.DataFrame(list(model_bnr_simple_results_class_test)).T,pd.DataFrame(list(model_bnr_simple_results_all_test)).T])
model_bnr_simple_all_df_test.columns = metric_columns
model_bnr_simple_all_df_test.index = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary', 'Total']
print('Simple CNN With Batch Normalization Regularization : Test Set Classification Performance')
model_bnr_simple_all_df_test

```

    Simple CNN With Batch Normalization Regularization : Test Set Classification Performance
    




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
      <td>0.881432</td>
      <td>0.972840</td>
      <td>0.924883</td>
      <td>405.0</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.900356</td>
      <td>0.843333</td>
      <td>0.870912</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.752650</td>
      <td>0.696078</td>
      <td>0.723260</td>
      <td>306.0</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.946667</td>
      <td>0.946667</td>
      <td>0.946667</td>
      <td>300.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.870276</td>
      <td>0.864729</td>
      <td>0.866430</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Consolidating all model evaluation metrics 
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
model_bnr_simple_model_list_test = []
model_bnr_simple_measure_list_test = []
model_bnr_simple_category_list_test = []
model_bnr_simple_value_list_test = []
model_bnr_simple_dataset_list_test = []

for i in range(3): 
    for j in range(5):
        model_bnr_simple_model_list_test.append('CNN_BNR_Simple')
        model_bnr_simple_measure_list_test.append(metric_columns[i])
        model_bnr_simple_category_list_test.append(model_bnr_simple_all_df_test.index[j])
        model_bnr_simple_value_list_test.append(model_bnr_simple_all_df_test.iloc[j,i])
        model_bnr_simple_dataset_list_test.append('Test')

model_bnr_simple_all_summary_test = pd.DataFrame(zip(model_bnr_simple_model_list_test,
                                                         model_bnr_simple_measure_list_test,
                                                         model_bnr_simple_category_list_test,
                                                         model_bnr_simple_value_list_test,
                                                         model_bnr_simple_dataset_list_test), 
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
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test = pd.concat([model_bnr_simple_all_summary_val,
                                                       model_bnr_simple_all_summary_test], 
                                                      ignore_index=True)

```


```python
##################################
# Consolidating all the precision
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_precision = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='Precision']
cnn_model_performance_comparison_val_test_precision_CNN_BNR_Simple_Validation = cnn_model_performance_comparison_val_test_precision[cnn_model_performance_comparison_val_test_precision['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_precision_CNN_BNR_Simple_Test = cnn_model_performance_comparison_val_test_precision[cnn_model_performance_comparison_val_test_precision['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the precision
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_precision_plot = pd.DataFrame({'CNN_BNR_Simple_Validation': cnn_model_performance_comparison_val_test_precision_CNN_BNR_Simple_Validation.values,
                                                                         'CNN_BNR_Simple_Test': cnn_model_performance_comparison_val_test_precision_CNN_BNR_Simple_Test.values},
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
      <th>CNN_BNR_Simple_Validation</th>
      <th>CNN_BNR_Simple_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.832317</td>
      <td>0.881432</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.962025</td>
      <td>0.900356</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.690647</td>
      <td>0.752650</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.845638</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.832657</td>
      <td>0.870276</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the precision
# for the selected model defined as
# simple CNN with batch normalization regularization
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


    
![png](output_241_0.png)
    



```python
##################################
# Consolidating all the recall
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_recall = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='Recall']
cnn_model_performance_comparison_val_test_recall_CNN_BNR_Simple_Validation = cnn_model_performance_comparison_val_test_recall[cnn_model_performance_comparison_val_test_recall['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_recall_CNN_BNR_Simple_Test = cnn_model_performance_comparison_val_test_recall[cnn_model_performance_comparison_val_test_recall['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the recall
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_recall_plot = pd.DataFrame({'CNN_BNR_Simple_Validation': cnn_model_performance_comparison_val_test_recall_CNN_BNR_Simple_Validation.values,
                                                                      'CNN_BNR_Simple_Test': cnn_model_performance_comparison_val_test_recall_CNN_BNR_Simple_Test.values},
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
      <th>CNN_BNR_Simple_Validation</th>
      <th>CNN_BNR_Simple_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.855799</td>
      <td>0.972840</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.863636</td>
      <td>0.843333</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.719101</td>
      <td>0.696078</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.865979</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.826129</td>
      <td>0.864729</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the recall
# for the selected model defined as
# simple CNN with batch normalization regularization
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


    
![png](output_244_0.png)
    



```python
##################################
# Consolidating all the fscore
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_fscore = cnn_model_performance_comparison_val_test[cnn_model_performance_comparison_val_test['Model.Metric']=='F-Score']
cnn_model_performance_comparison_val_test_fscore_CNN_BNR_Simple_Validation = cnn_model_performance_comparison_val_test_fscore[cnn_model_performance_comparison_val_test_fscore['Data.Set']=='Validation'].loc[:,"Metric.Value"]
cnn_model_performance_comparison_val_test_fscore_CNN_BNR_Simple_Test = cnn_model_performance_comparison_val_test_fscore[cnn_model_performance_comparison_val_test_fscore['Data.Set']=='Test'].loc[:,"Metric.Value"]

```


```python
##################################
# Combining all the fscore
# model performance measures
# for the selected model defined as
# simple CNN with batch normalization regularization
##################################
cnn_model_performance_comparison_val_test_fscore_plot = pd.DataFrame({'CNN_BNR_Simple_Validation': cnn_model_performance_comparison_val_test_fscore_CNN_BNR_Simple_Validation.values,
                                                                      'CNN_BNR_Simple_Test': cnn_model_performance_comparison_val_test_fscore_CNN_BNR_Simple_Test.values},
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
      <th>CNN_BNR_Simple_Validation</th>
      <th>CNN_BNR_Simple_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No Tumor</th>
      <td>0.843895</td>
      <td>0.924883</td>
    </tr>
    <tr>
      <th>Glioma</th>
      <td>0.910180</td>
      <td>0.870912</td>
    </tr>
    <tr>
      <th>Meningioma</th>
      <td>0.704587</td>
      <td>0.723260</td>
    </tr>
    <tr>
      <th>Pituitary</th>
      <td>0.855688</td>
      <td>0.946667</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.828587</td>
      <td>0.866430</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting all the fscore
# for the selected model defined as
# simple CNN with batch normalization regularization
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


    
![png](output_247_0.png)
    



```python
##################################
# Evaluating the model
# for a complex CNN with dropout and batch normalization regularization
# on the independent test set
##################################
model_cdrbnr_complex_y_pred_test = model_cdrbnr_complex.predict(test_gen)
```

    [1m41/41[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 103ms/step
    


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


    
![png](output_249_0.png)
    



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
# for the selected model defined as
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
# for the selected model defined as
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
# for the selected model defined as
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
# for the selected model defined as
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
# for the selected model defined as
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


    
![png](output_255_0.png)
    



```python
##################################
# Consolidating all the recall
# model performance measures
# for the selected model defined as
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
# for the selected model defined as
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
# for the selected model defined as
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


    
![png](output_258_0.png)
    



```python
##################################
# Consolidating all the fscore
# model performance measures
# for the selected model defined as
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
# for the selected model defined as
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
# for the selected model defined as
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


    
![png](output_261_0.png)
    


### 1.6.9 Model Inference <a class="anchor" id="1.6.9"></a>

1. The gradient-weighted class activation map for  the first convolutional layer of the selected model - **Complex CNN Model With Dropout and Batch Normalization Regularization** highlighted generic image features that lead to the activation of the different image categories.
    * Images identified with <span style="color: #FF0000">CLASS: No Tumor</span> had the following characteristics:
        * Smooth and symmetric brain structures without abrupt changes in intensity
        * High symmetry in both hemispheres
    * Images identified with <span style="color: #FF0000">CLASS: Glioma</span> had the following characteristics:
        * Irregular and poorly defined edges with heterogeneous intensity in the parenchymal brain tissue, particularly white matter (frontal or temporal lobes)
        * Asymmetric disruptions, particularly in deep structures like the thalamus
    * Images identified with <span style="color: #FF0000">CLASS: Meningioma</span> had the following characteristics:
        * Clear and sharp boundaries with uniform or slightly heterogeneous intensity from the meninges, commonly at the convexity or skull base
        * Localized asymmetry near the meninges
    * Images identified with <span style="color: #FF0000">CLASS: Pituitary</span> had the following characteristics:
        * Smooth or mildly irregular boundaries within the pituitary region
        * Localized asymmetry near the pituitary region
2. The gradient-weighted class activation map for  the second convolutional layer of the selected model - **Complex CNN Model With Dropout and Batch Normalization Regularization** highlighted specific image features that lead to the activation of the different image categories.
    * Images identified with <span style="color: #FF0000">CLASS: No Tumor</span> had the following characteristics:
        * Normal brain morphology without extrinsic masses
    * Images identified with <span style="color: #FF0000">CLASS: Glioma</span> had the following characteristics:
        * Irregular shapes with infiltrative extensions into surrounding tissues
    * Images identified with <span style="color: #FF0000">CLASS: Meningioma</span> had the following characteristics:
        * Dome-shaped or spherical, often attached to the meninges, commonly at the convexity or skull base
    * Images identified with <span style="color: #FF0000">CLASS: Pituitary</span> had the following characteristics:
        * Ovoid or flattened tumor localized in the pituitary region
3. The gradient-weighted class activation map for  the third and final convolutional layer of the selected model - **Complex CNN Model With Dropout and Batch Normalization Regularization** highlighted detailed image features that lead to the activation of the different image categories.
    * Images identified with <span style="color: #FF0000">CLASS: No Tumor</span> had the following characteristics:
        * Normal anatomical landmarks without any mass effect
    * Images identified with <span style="color: #FF0000">CLASS: Glioma</span> had the following characteristics:
        * Hyper-intense masses in the parenchymal brain tissue, particularly white matter (frontal or temporal lobes) 
        * Significant mass effect including ventricular compression or midline shift
    * Images identified with <span style="color: #FF0000">CLASS: Meningioma</span> had the following characteristics:
        * Hyper-intense masses arising from the meninges, commonly at the convexity or skull base 
        * Significant mass effect including ventricular compression or midline shift
    * Images identified with <span style="color: #FF0000">CLASS: Pituitary</span> had the following characteristics:
        * Hyper-intense masses exclusively in the pituitary region 
        * Localized deformation at the pituitary region



```python
##################################
# Gathering the actual and predicted classes
# from the selected CNN model defined as
# complex CNN with dropout and batch normalization regularization
##################################
model_cdrbnr_complex_predictions_test = np.array(list(map(lambda x: np.argmax(x), model_cdrbnr_complex_y_pred_test)))
model_cdrbnr_complex_y_true_test = test_gen.classes
```


```python
##################################
# Consolidating the actual and predicted classes
# from the selected CNN model defined as
# complex CNN with dropout and batch normalization regularization
##################################
class_indices = test_gen.class_indices
indices = {v:k for k,v in class_indices.items()}
filenames = test_gen.filenames
test_gen_df = pd.DataFrame()
test_gen_df['FileName'] = filenames
test_gen_df['Actual_Category'] = model_cdrbnr_complex_y_true_test
test_gen_df['Predicted_Category'] = model_cdrbnr_complex_predictions_test
test_gen_df['Actual_Category'] = test_gen_df['Actual_Category'].apply(lambda x: indices[x])
test_gen_df['Predicted_Category'] = test_gen_df['Predicted_Category'].apply(lambda x: indices[x])
test_gen_df.loc[test_gen_df['Actual_Category']==test_gen_df['Predicted_Category'],'Matched_Category_Prediction'] = True
test_gen_df.loc[test_gen_df['Actual_Category']!=test_gen_df['Predicted_Category'],'Matched_Category_Prediction'] = False
test_gen_df.head(10)
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
      <th>FileName</th>
      <th>Actual_Category</th>
      <th>Predicted_Category</th>
      <th>Matched_Category_Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>notumor\Te-noTr_0000.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>notumor\Te-noTr_0001.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>notumor\Te-noTr_0002.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>notumor\Te-noTr_0003.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>notumor\Te-noTr_0004.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>notumor\Te-noTr_0005.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>notumor\Te-noTr_0006.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>notumor\Te-noTr_0007.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>notumor\Te-noTr_0008.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>notumor\Te-noTr_0009.jpg</td>
      <td>notumor</td>
      <td>notumor</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating image samples
# from the test set
##################################
test_gen_df = test_gen_df.sample(frac=1, replace=False, random_state=123).reset_index(drop=True)
```


```python
##################################
# Defining a function
# to load the sampled images
##################################
img_size=227
def readImage(path):
    img = load_img(path,color_mode="grayscale", target_size=(img_size,img_size))
    img = img_to_array(img)
    img = img/255.    
    return img
```


```python
##################################
# Defining a function
# to display the sampled images
# with the actual and predicted categories
##################################
base_path = (os.path.join("..", DATASETS_FINAL_TEST_PATH))
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize = (20 , 20))
    n = 0
    for i in range(15):
        n+=1
        plt.subplot(5 , 5, n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)
        image = readImage(f"{base_path}\\{temp_df.FileName[i]}")
        plt.imshow(image)
        plt.title(f'A: {temp_df.Actual_Category[i]} P: {temp_df.Predicted_Category[i]}')
```


```python
##################################
# Display sample images with matched
# actual and predicted categories
##################################
display_images(test_gen_df[test_gen_df['Matched_Category_Prediction']==True])
```


    
![png](output_268_0.png)
    



```python
##################################
# Display sample images with mismatched
# actual and predicted categories
##################################
display_images(test_gen_df[test_gen_df['Matched_Category_Prediction']!=True])
```


    
![png](output_269_0.png)
    



```python
##################################
# Recreating the CNN model defined as
# complex CNN with dropout and batch normalization regularization
# using the Functional API structure
##################################

##################################
# Defining the input layer
##################################
fmodel_input_layer = Input(shape=(227, 227, 1), name="input_layer")

##################################
# Using the layers from the Sequential model
# as functions in the Functional API
##################################
set_seed()
fmodel_conv2d_layer = model_cdrbnr_complex.layers[0](fmodel_input_layer) # Conv2D layer
fmodel_maxpooling2d_layer = model_cdrbnr_complex.layers[1](fmodel_conv2d_layer) # MaxPooling2D layer
fmodel_conv2d_1_layer = model_cdrbnr_complex.layers[2](fmodel_maxpooling2d_layer) # Conv2D layer
fmodel_maxpooling2d_1_layer = model_cdrbnr_complex.layers[3](fmodel_conv2d_1_layer) # MaxPooling2D layer
fmodel_conv2d_2_layer = model_cdrbnr_complex.layers[4](fmodel_maxpooling2d_1_layer) # Conv2D layer
fmodel_batchnormalization_layer = model_cdrbnr_complex.layers[5](fmodel_conv2d_2_layer) # Batch Normalization layer
fmodel_activation_layer = model_cdrbnr_complex.layers[6](fmodel_batchnormalization_layer) # Activation layer
fmodel_maxpooling2d_2_layer = model_cdrbnr_complex.layers[7](fmodel_activation_layer) # MaxPooling2D layer
fmodel_flatten_layer = model_cdrbnr_complex.layers[8](fmodel_maxpooling2d_2_layer) # Flatten layer
fmodel_dense_layer = model_cdrbnr_complex.layers[9](fmodel_flatten_layer) # Dense layer (128 units)
fmodel_dropout_layer = model_cdrbnr_complex.layers[10](fmodel_dense_layer) # Dropout layer
fmodel_output_layer = model_cdrbnr_complex.layers[11](fmodel_dropout_layer) # Dense layer (num_classes units)

##################################
# Creating the Functional API model
##################################
model_cdrbnr_complex_functional_api = Model(inputs=fmodel_input_layer, outputs=fmodel_output_layer, name="model_cdrbnr_complex_fapi")

##################################
# Compiling the Functional API model
# with the same parameters
##################################
set_seed()
model_cdrbnr_complex_functional_api.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Displaying the model summary
# for CNN with dropout regularization
##################################
print(model_cdrbnr_complex_functional_api.summary())

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_cdrbnr_complex_fapi"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">227</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
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
# Creating a gradient model for the
# gradient class activation map
# of the first convolutional layer
##################################
grad_model_first_conv2d = Model(inputs=fmodel_input_layer, outputs=[fmodel_conv2d_layer, fmodel_output_layer], name="model_cdrbnr_complex_fapi_first_conv2d")
set_seed()
grad_model_first_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the first convolutional layer
##################################
def make_gradcam_heatmap(img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_first_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds
    
```


```python
##################################
# Defining a function
# to colorize the generated heatmap
# and superimpose on the actual image
##################################
def gradCAMImage(image):
    path = (os.path.join("..", DATASETS_FINAL_TEST_PATH, image))
    img = readImage(path)
    img = np.expand_dims(img,axis=0)
    heatmap, preds = make_gradcam_heatmap(img)

    img = load_img(path)
    img = img_to_array(img)
    heatmap = np.uint8(255 * heatmap)

    jet = plt.colormaps["turbo"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.80 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img
    
```


```python
##################################
# Defining a function to consolidate
# the gradient class activation maps
# for a subset of sampled images
##################################
def gradcam_of_images(correct_class):
    grad_images = []
    title = []
    temp_df = test_gen_df[test_gen_df['Matched_Category_Prediction']==correct_class]
    temp_df = temp_df.reset_index(drop=True)
    for i in range(15):
        image = temp_df.FileName[i]
        grad_image = gradCAMImage(image)
        grad_images.append(grad_image)
        title.append(f"A: {temp_df.Actual_Category[i]} P: {temp_df.Predicted_Category[i]}")

    return grad_images, title
    
```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the first convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
matched_categories, matched_categories_titles = gradcam_of_images(correct_class=True)

```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the first convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
mismatched_categories, mismatched_categories_titles = gradcam_of_images(correct_class=False)

```


```python
##################################
# Defining a function to display
# the consolidated gradient class activation maps
# for a subset of sampled images
##################################
def display_heatmaps(classified_images, titles):
    plt.figure(figsize = (20 , 20))
    n = 0
    for i in range(15):
        n+=1
        plt.subplot(5 , 5, n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)
        plt.imshow(classified_images[i])
        plt.title(titles[i])
    plt.show()
    
```


```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the first convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
display_heatmaps(matched_categories, matched_categories_titles)

```


    
![png](output_278_0.png)
    



```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the first convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
display_heatmaps(mismatched_categories, mismatched_categories_titles)

```


    
![png](output_279_0.png)
    



```python
##################################
# Creating a gradient model for the
# gradient class activation map
# of the second convolutional layer
##################################
grad_model_second_conv2d = Model(inputs=fmodel_input_layer, outputs=[fmodel_conv2d_1_layer, fmodel_output_layer], name="model_cdrbnr_complex_fapi_second_conv2d")
set_seed()
grad_model_second_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the second convolutional layer
##################################
def make_gradcam_heatmap(img_array, pred_index=None):    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_second_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds
    
```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the second convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
matched_categories, matched_categories_titles = gradcam_of_images(correct_class=True)

```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the second convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
mismatched_categories, mismatched_categories_titles = gradcam_of_images(correct_class=False)

```


```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the second convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
display_heatmaps(matched_categories, matched_categories_titles)

```


    
![png](output_284_0.png)
    



```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the second convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
display_heatmaps(mismatched_categories, mismatched_categories_titles)

```


    
![png](output_285_0.png)
    



```python
##################################
# Creating a gradient model for the
# gradient class activation map
# of the third convolutional layer
##################################
grad_model_third_conv2d = Model(inputs=fmodel_input_layer, outputs=[fmodel_conv2d_2_layer, fmodel_output_layer], name="model_cdrbnr_complex_fapi_third_conv2d")
set_seed()
grad_model_third_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

```


```python
##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the third convolutional layer
##################################
def make_gradcam_heatmap(img_array, pred_index=None):    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_third_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds
    
```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the third convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
matched_categories, matched_categories_titles = gradcam_of_images(correct_class=True)

```


```python
##################################
# Consolidating the gradient class activation maps
# from the output of the third convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
mismatched_categories, mismatched_categories_titles = gradcam_of_images(correct_class=False)

```


```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the third convolutional layer
# for the subset of sampled images
# with matched actual and predicted categories
##################################
display_heatmaps(matched_categories, matched_categories_titles)

```


    
![png](output_290_0.png)
    



```python
##################################
# Displaying the consolidated 
# gradient class activation maps
# from the output of the third convolutional layer
# for the subset of sampled images
# with mismatched actual and predicted categories
##################################
display_heatmaps(mismatched_categories, mismatched_categories_titles)

```


    
![png](output_291_0.png)
    


## 1.7 Predictive Model Deployment Using Streamlit and Streamlit Community Cloud <a class="anchor" id="1.7"></a>

### 1.7.1 Model Application Programming Interface Code Development <a class="anchor" id="1.7.1"></a>

1. A model prediction application code in Python was developed to:
    * randomly sample and preprocess an image and assign as a test case
    * generate the RGB and Grad-CAM visualization plots for the test case
    * estimate the image class probabilities for the test case
    * predict the image class for the test case
2. The model prediction application code was saved in a repository that was eventually cloned for uploading to Streamlit Community Cloud.
   

![ModelDeployment3_ModelPredictionApplicationCode.png](994298b0-5c3a-4971-bc16-8ef8a02f37b2.png)

### 1.7.2 User Interface Application Code Development <a class="anchor" id="1.7.2"></a>

1. A user interface application code in Python was developed to:
    * randomly sample and preprocess an image and assign as a test case
    * generate the RGB and Grad-CAM visualization plots for the test case
    * estimate the image class probabilities for the test case
    * predict the image class for the test case
2. The user interface application code was saved in a repository that was eventually cloned for uploading to Streamlit Community Cloud.


![ModelDeployment3_UserInterfaceApplicationCode.png](58a4212c-63c4-4f65-a046-6c03a8ea9040.png)

### 1.7.3 Web Application <a class="anchor" id="1.7.3"></a>

1. The prediction model was deployed using a web application hosted at [<mark style="background-color: #CCECFF"><b>Streamlit</b></mark>](https://brain-mri-image-classification.streamlit.app/).
2. The user interface input consists of the following:
    * First action button to:
        * randomly sample an MR image as a test case
        * conduct image preprocessing
        * display the RGB channels
        * activates the second action button
    * Second action button to:
        * load fitted CNN model
        * estimate image class probabilities
        * predict class categories
        * perform the Grad-CAM computation
        * display the Grad-CAM visualization for all convolutional layers
        * render test case prediction summary
3. The user interface ouput consists of the following:
    * RGB plots to:
        * provide a baseline visualization of the test case by channel
    * Grad-CAM plots to:
        * present insights into how the model progressively learns features (from low-level to high-level) for each convolutional layer, aiding in understanding spatial and hierarchical representation
        * highlight image regions that influenced the model's decision the most allowing to verify whether the model focuses on relevant areas
    * summary table to:
        * indicate if the model prediction matches the ground truth
        * present the estimated class probabilities and predicted class  for the test case


![ModelDeployment3_WebApplication.png](d0cfc2f9-9981-4d41-8756-8af64f34a00a.png)

# 2. Summary <a class="anchor" id="Summary"></a>

![ModelDeployment3_Summary_0.png](4c1ee28e-3042-4e0a-8e4f-248984117419.png)

![ModelDeployment3_Summary_1.png](58f060d7-c4d1-40ea-baef-b8b62e8e02ac.png)

![ModelDeployment3_Summary_2.png](4febf07c-4eed-4935-81b1-377e660e40b3.png)

![ModelDeployment3_Summary_3.png](36b9dced-433f-43fe-b5d0-3086eb32706c.png)

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by Francois Chollet
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
* **[Python Library API]** [scipy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [statsmodels](https://www.statsmodels.org/stable/index.html) by statsmodels Team
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
* **[Kaggle Project]** [Glioma 19 Radiography Data - EDA and CNN Model](https://www.kaggle.com/code/jnegrini/glioma-19-radiography-data-eda-and-cnn-model) by Juliana Negrini De Araujo (Kaggle)
* **[Kaggle Project]** [Pneumonia Detection using CNN (92.6% Accuracy)](https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy) by Madhav Mathur (Kaggle)
* **[Kaggle Project]** [Glioma Detection from CXR Using Explainable CNN](https://www.kaggle.com/code/sid321axn/glioma-detection-from-cxr-using-explainable-cnn) by Manu Siddhartha (Kaggle)
* **[Kaggle Project]** [Class Activation Mapping for glioma-19 CNN](https://www.kaggle.com/code/amyjang/class-activation-mapping-for-glioma-19-cnn) by Amy Zhang (Kaggle)
* **[Kaggle Project]** [CNN mri glioma Classification](https://www.kaggle.com/code/gabrielmino/cnn-mri-glioma-classification) by Gabriel Mino (Kaggle)
* **[Kaggle Project]** [Detecting-glioma-19-Images | CNN](https://www.kaggle.com/code/felipeoliveiraml/detecting-glioma-19-images-cnn) by Felipe Oliveira (Kaggle)
* **[Kaggle Project]** [Detection of glioma Positive Cases using DL](https://www.kaggle.com/code/sana306/detection-of-glioma-positive-cases-using-dl) by Sana Shaikh (Kaggle)
* **[Kaggle Project]** [Deep Learning and Transfer Learning on glioma-19](https://www.kaggle.com/code/digvijayyadav/deep-learning-and-transfer-learning-on-glioma-19) by Digvijay Yadav (Kaggle)
* **[Kaggle Project]** [X-ray Detecting Using CNN](https://www.kaggle.com/code/shivan118/x-ray-detecting-using-cnn) by Shivan Kumar (Kaggle)
* **[Kaggle Project]** [Classification of glioma-19 using CNN](https://www.kaggle.com/code/islamselim/classification-of-glioma-19-using-cnn) by Islam Selim (Kaggle)
* **[Kaggle Project]** [Glioma-19 - Revisiting Pneumonia Detection](https://www.kaggle.com/code/pcbreviglieri/glioma-19-revisiting-pneumonia-detection) by Paulo Breviglieri (Kaggle)
* **[Kaggle Project]** [Multi-Class X-ray glioma19 Classification-94% Accurary](https://www.kaggle.com/code/derrelldsouza/multi-class-x-ray-glioma19-classification-94-acc) by Quadeer Shaikh (Kaggle)
* **[Kaggle Project]** [Grad-CAM: What Do CNNs See?](https://www.kaggle.com/code/quadeer15sh/grad-cam-what-do-cnns-see) by Derrel Souza (Kaggle)
* **[GitHub Project]** [Grad-CAM](https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb) by Ismail Uddin (GitHub)
* **[Publication]** [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) by Yann LeCun, Leon Bottou, Yoshua Bengio and Patrick Haffner (Proceedings of the IEEE)
* **[Publication]** [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) by Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva and Antonio Torralba (Computer Vision and Pattern Recognition)
* **[Publication]** [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) by Ramprasaath Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh and Dhruv Batra (Computer Vision and Pattern Recognition)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)
* **[Course]** [DeepLearning.AI Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) by DeepLearning.AI Team (Coursera)
* **[Course]** [DeepLearning.AI Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by DeepLearning.AI Team (Coursera)
* **[Course]** [DeepLearning.AI TensorFlow: Advanced Techniques Specialization](https://www.coursera.org/specializations/tensorflow-advanced-techniques) by DeepLearning.AI Team (Coursera)
* **[Course]** [DeepLearning.AI TensorFlow: Data and Deployment Specialization](https://www.coursera.org/specializations/tensorflow-data-and-deployment) by DeepLearning.AI Team (Coursera)
* **[Course]** [DataCamp Machine Learning Scientist in Python Track](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Image Processing in Python Track](https://app.datacamp.com/learn/skill-tracks/image-processing) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Keras Fundamentals Track](https://app.datacamp.com/learn/skill-tracks/keras-fundamentals) by DataCamp Team (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

