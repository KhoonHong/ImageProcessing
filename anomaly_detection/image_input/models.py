from django.conf import settings
from django.db import models
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    image_url = models.CharField(max_length=255, null=True, blank=True)
    image_name = models.CharField(max_length=255, null=True, blank=True)
    image_label = models.CharField(max_length=255, null=True, blank=True, default='NULL')

    def save(self, *args, **kwargs):
        super(UploadedImage, self).save(*args, **kwargs)  # First, save the image
        # After saving the image, its file path is accessible through self.image.url
        # Create or update the image_url field
        self.image_url = settings.MEDIA_URL + str(self.image)
        self.image_name = str(self.image)
        super(UploadedImage, self).save(*args, **kwargs)  # Save the instance again after updating the image_url

    def update_label(self, new_label):
        self.image_label = new_label
        self.save()

def create_autoencoder():
    input_img = Input(shape=(128, 128, 3)) # assuming your images are 128x128x3; adjust if different
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder(data):
    autoencoder = create_autoencoder()
    # Assuming data is normalized to [0,1]
    autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_data=(data, data))
    autoencoder.save('autoencoder_model.h5')