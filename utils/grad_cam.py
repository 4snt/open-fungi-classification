import tensorflow as tf
from keras import backend as K
import numpy as np

def grad_cam(model, image, cls, layer_name):
    with tf.GradientTape() as tape:
        tape.watch(model.input)
        y_c = model.output[0, cls]
        conv_output = model.get_layer(layer_name).output

    grads = tape.gradient(y_c, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()
