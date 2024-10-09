# src/utils/utils.py
import tensorflow as tf

# define loss function
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))