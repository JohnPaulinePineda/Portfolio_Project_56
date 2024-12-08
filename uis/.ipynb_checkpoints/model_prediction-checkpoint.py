##################################
# Loading Python Libraries
##################################
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

##################################
# Defining file paths
##################################
MODELS_PATH = r"models"

##################################
# Loading the final classification model
# from the MODELS_PATH
##################################
final_cnn_model = load_model(os.path.join("..",MODELS_PATH, "cdrbnr_complex_best_model.keras"))
