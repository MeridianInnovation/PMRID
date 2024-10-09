# src/utils/utils.py
import yaml
import tensorflow as tf
import os

# define loss function
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))