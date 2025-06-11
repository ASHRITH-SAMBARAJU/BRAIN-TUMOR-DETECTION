# utils/gradcam.py

import numpy as np
import cv2
import tensorflow as tf

def generate_gradcam(model, image, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.uint8(255 * image)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed
