from .models import create_autoencoder
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('autoencoder_model.h5')

def compute_anomaly_score(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    reconstructed = model.predict(img_array)
    
    error = np.mean(np.square(img_array - reconstructed))
    return error
