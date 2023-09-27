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
