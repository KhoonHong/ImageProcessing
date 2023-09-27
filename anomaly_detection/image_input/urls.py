from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_view, name='upload_view'),
    path('delete/', views.delete_images, name='delete_images'),
    path('image_reg/', views.image_reg, name='image_reg'),
    path('imageReg_featureDetect/', views.feature_detect, name='feature_detect'),
    path('imageRegAndAlign/', views.image_reg_align, name='image_reg_align'),
    path('update_label/', views.UpdateLabelView.as_view(), name='update_label'),
    path('anomaly_detection/', views.anomaly_detection, name='anomaly_detection'),
    path('morphological_methods/', views.morphological_methods, name='morphological_methods'),
    path('morphological_methods_demo/', views.morph_demo, name='morph_demo'),
    path('output/', views.output, name='output'),
    path('morphMethodsApplyAll/', views.morphMethodsApplyAll, name='morphMethodsApplyAll'),
    path('anomalyDetectionDemo/', views.anomalyDetection, name='anomalyDetection'),
    path('image_output/', views.image_output, name='image_output'),
]

