from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView, ListView, CreateView, UpdateView, DeleteView
from django.contrib import messages
from django.urls import reverse_lazy
from analytics.models import Dataset
from analytics.services import DataAnalysisService
from .forms import DatasetUploadForm
import pandas as pd
import os

class UploadDatasetView(LoginRequiredMixin, CreateView):
    model = Dataset
    form_class = DatasetUploadForm
    template_name = 'data_upload/upload.html'
    success_url = reverse_lazy('analytics:datasets')
    
    def form_valid(self, form):
        form.instance.owner = self.request.user
        
        # Sauvegarder le dataset
        response = super().form_valid(form)
        
        # Analyser le fichier pour obtenir les métadonnées
        try:
            self._analyze_file_metadata(self.object)
            messages.success(
                self.request, 
                f'Dataset "{self.object.name}" uploadé avec succès ! '
                f'{self.object.rows_count} lignes et {self.object.columns_count} colonnes détectées.'
            )
        except Exception as e:
            messages.warning(
                self.request,
                f'Dataset uploadé mais erreur lors de l\'analyse des métadonnées: {str(e)}'
            )
        
        return response
    
    def _analyze_file_metadata(self, dataset):
        """Analyser le fichier pour extraire les métadonnées"""
        try:
            service = DataAnalysisService()
            df = service.load_dataset(dataset)
            
            # Mettre à jour les métadonnées
            dataset.rows_count = len(df)
            dataset.columns_count = len(df.columns)
            dataset.file_size = os.path.getsize(dataset.file_path.path)
            dataset.save()
            
        except Exception as e:
            raise Exception(f"Erreur lors de l'analyse du fichier: {str(e)}")

class DatasetPreviewView(LoginRequiredMixin, DetailView):
    """Vue pour prévisualiser un dataset avant confirmation"""
    model = Dataset
    template_name = 'data_upload/preview.html'
    context_object_name = 'dataset'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        try:
            service = DataAnalysisService()
            preview_data = service.get_dataset_preview(self.object, rows=20)
            context['preview'] = preview_data
        except Exception as e:
            messages.error(self.request, f'Erreur lors du chargement de la prévisualisation: {str(e)}')
            context['preview'] = None
        
        return context