from django.urls import path
from . import views

app_name = 'data_upload'

urlpatterns = [
    path('upload/', views.UploadDatasetView.as_view(), name='upload'),
    path('preview/<int:pk>/', views.DatasetPreviewView.as_view(), name='preview'),
]