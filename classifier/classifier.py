#functions.py

#CLASSIFIER_PATH = 'BirdNET/checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP32.tflite'
import tensorflow as tf


def load_model(path):
    classifier = tf.keras.models.load_model(path)
    return classifier
