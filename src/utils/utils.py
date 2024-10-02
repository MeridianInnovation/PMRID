# src/utils/utils.py
import yaml
import tensorflow as tf
import os 

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# define loss function
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))