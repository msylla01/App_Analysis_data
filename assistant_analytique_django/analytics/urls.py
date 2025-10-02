from django.urls import path
from . import views  # Importer seulement views, pas chat_views

app_name = 'analytics'

urlpatterns = [
    # Dashboard et analyses existantes
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('datasets/', views.DatasetListView.as_view(), name='datasets'),
    path('datasets/<int:pk>/', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/<int:pk>/analyze/', views.CreateAnalysisView.as_view(), name='create_analysis'),
    path('analyses/', views.AnalysisListView.as_view(), name='analyses'),
    path('analyses/<int:pk>/', views.AnalysisDetailView.as_view(), name='analysis_detail'),
    path('analyses/<int:pk>/delete/', views.AnalysisDeleteView.as_view(), name='analysis_delete'),
    
    # Visualisations existantes
    path('visualizations/', views.VisualizationListView.as_view(), name='visualizations'),
    path('visualizations/<int:pk>/', views.VisualizationDetailView.as_view(), name='visualization_detail'),
    path('visualizations/create/', views.VisualizationCreateView.as_view(), name='visualization_create'),
    path('visualizations/<int:pk>/edit/', views.VisualizationUpdateView.as_view(), name='visualization_update'),
    path('visualizations/<int:pk>/delete/', views.VisualizationDeleteView.as_view(), name='visualization_delete'),
    
    # API endpoints pour visualisations
    path('visualizations/<int:pk>/image/', views.VisualizationImageView.as_view(), name='visualization_image'),
    path('visualizations/<int:pk>/stats/', views.VisualizationStatsView.as_view(), name='visualization_stats'),
    path('visualizations/<int:pk>/export/', views.VisualizationExportView.as_view(), name='visualization_export'),
    
    # Visualisations publiques
    path('public/visualizations/<int:pk>/', views.PublicVisualizationDetailView.as_view(), name='public_visualization'),
    
    # === URLS POUR LE CHAT (maintenant dans views.py) ===
    # Gestion des sessions de chat
    path('chat/', views.ChatSessionListView.as_view(), name='chat_sessions'),
    path('chat/create/', views.ChatSessionCreateView.as_view(), name='chat_create'),
    path('chat/<uuid:pk>/', views.ChatSessionDetailView.as_view(), name='chat_session_detail'),
    path('chat/<uuid:pk>/delete/', views.ChatSessionDeleteView.as_view(), name='chat_session_delete'),
    
    # API pour les messages
    path('chat/<uuid:session_id>/message/', views.ChatMessageCreateView.as_view(), name='chat_message_create'),
    path('chat/<uuid:session_id>/dataset/', views.ChatSessionUpdateDatasetView.as_view(), name='chat_update_dataset'),
    
    # Feedback et export
    path('chat/message/<uuid:message_id>/feedback/', views.chat_message_feedback, name='chat_message_feedback'),
    path('chat/<uuid:session_id>/export/', views.ChatExportView.as_view(), name='chat_export'),
    
    # Analytics du chat
    path('chat/analytics/', views.ChatAnalyticsView.as_view(), name='chat_analytics'),
]