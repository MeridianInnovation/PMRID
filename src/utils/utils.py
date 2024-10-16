# src/utils/utils.py
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# define loss function

# l1 loss function for denoising
@register_keras_serializable(package='Custom', name='l1_loss')
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# ssim loss function for denoising
@register_keras_serializable(package='Custom', name='ssim_loss')
def ssim_loss(y_true, y_pred):
    ssim_values = tf.image.ssim(y_true, y_pred, max_val=255)
    return 1 - ssim_values