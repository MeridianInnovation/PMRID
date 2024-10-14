# src/utils/utils.py
import yaml
import tensorflow as tf
from tf.keras.saving import register_keras_serializable

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# define loss function
@register_keras_serializable(package='Custom', name='loss_function')
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))