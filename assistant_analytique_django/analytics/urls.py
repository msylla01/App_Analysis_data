from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    # Dashboard et analyses
    path('', views.DashboardView.as_view(), name='dashboard'),
    
    # Datasets
    path('datasets/', views.DatasetListView.as_view(), name='datasets'),
    path('datasets/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/<int:pk>/analyze/', views.CreateAnalysisView.as_view(), name='create_analysis'),
    
    # Analyses
    path('analyses/', views.AnalysisListView.as_view(), name='analyses'),
    path('analyses/<int:pk>/', views.AnalysisDetailView.as_view(), name='analysis_detail'),
    path('analyses/<int:pk>/delete/', views.AnalysisDeleteView.as_view(), name='analysis_delete'),
    
    # Visualisations
    path('visualizations/', views.VisualizationListView.as_view(), name='visualizations'),
    path('visualizations/<int:pk>/', views.VisualizationDetailView.as_view(), name='visualization_detail'),
    path('visualizations/create/', views.VisualizationCreateView.as_view(), name='visualization_create'),
    path('visualizations/<int:pk>/edit/', views.VisualizationUpdateView.as_view(), name='visualization_update'),
    path('visualizations/<int:pk>/delete/', views.VisualizationDeleteView.as_view(), name='visualization_delete'),
    
    # API endpoints
    path('visualizations/<int:pk>/image/', views.VisualizationImageView.as_view(), name='visualization_image'),
    path('visualizations/<int:pk>/stats/', views.VisualizationStatsView.as_view(), name='visualization_stats'),
    path('visualizations/<int:pk>/export/', views.VisualizationExportView.as_view(), name='visualization_export'),
    
    # Visualisations publiques
    path('public/visualizations/<int:pk>/', views.PublicVisualizationDetailView.as_view(), name='public_visualization'),
]