from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth import get_user_model
import json

User = get_user_model()

class Dataset(models.Model):
    DATASET_TYPES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('json', 'JSON'),
        ('api', 'API'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file_path = models.FileField(upload_to='datasets/')
    dataset_type = models.CharField(max_length=10, choices=DATASET_TYPES)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_public = models.BooleanField(default=False)
    
    # Métadonnées du dataset
    rows_count = models.IntegerField(null=True, blank=True)
    columns_count = models.IntegerField(null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)  # en bytes
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name

class Analysis(models.Model):
    ANALYSIS_TYPES = [
        ('descriptive', 'Analyse Descriptive'),
        ('correlation', 'Analyse de Corrélation'),
        ('regression', 'Régression'),
        ('clustering', 'Clustering'),
        ('timeseries', 'Série Temporelle'),
        ('custom', 'Personnalisée'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'En attente'),
        ('running', 'En cours'),
        ('completed', 'Terminée'),
        ('failed', 'Échouée'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPES)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Configuration de l'analyse
    parameters = models.JSONField(default=dict, blank=True)
    
    # Résultats
    results = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.get_analysis_type_display()}"
    
class Visualization(models.Model):
    CHART_TYPES = [
        ('line', 'Graphique en ligne'),
        ('bar', 'Graphique en barres'),
        ('scatter', 'Nuage de points'),
        ('pie', 'Graphique circulaire'),
        ('doughnut', 'Graphique en anneau'),
        ('radar', 'Graphique radar'),
        ('histogram', 'Histogramme'),
        ('heatmap', 'Carte de chaleur'),
        ('box', 'Boîte à moustaches'),
    ]
    
    STATUS_CHOICES = [
        ('draft', 'Brouillon'),
        ('published', 'Publié'),
        ('archived', 'Archivé'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    analysis = models.ForeignKey('Analysis', on_delete=models.CASCADE, related_name='visualizations')
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES)
    
    # Configuration du graphique
    chart_config = models.JSONField(default=dict)
    chart_data = models.JSONField(default=dict)
    
    # Métadonnées
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='published')
    is_public = models.BooleanField(default=False)
    is_featured = models.BooleanField(default=False)
    
    # Interactions
    views_count = models.PositiveIntegerField(default=0)
    downloads_count = models.PositiveIntegerField(default=0)
    
    # Paramètres d'affichage
    width = models.PositiveIntegerField(default=800)
    height = models.PositiveIntegerField(default=500)
    background_color = models.CharField(max_length=7, default='#ffffff')
    
    # Tags pour la recherche
    tags = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        db_table = 'analytics_visualization'  # Forcer le nom de table
    
    def __str__(self):
        return f"{self.title} ({self.get_chart_type_display()})"
    
    def get_chart_data_json(self):
        """Retourne les données du graphique en JSON sérialisé"""
        return json.dumps(self.chart_data, ensure_ascii=False)
    
    def get_chart_config_json(self):
        """Retourne la configuration du graphique en JSON sérialisé"""
        return json.dumps(self.chart_config, ensure_ascii=False)
    
    def increment_views(self):
        """Incrémenter le nombre de vues"""
        self.views_count += 1
        self.save(update_fields=['views_count'])
    
    def increment_downloads(self):
        """Incrémenter le nombre de téléchargements"""
        self.downloads_count += 1
        self.save(update_fields=['downloads_count'])
    
    def get_tags_list(self):
        """Retourner la liste des tags"""
        if self.tags:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
        return []
    
    def set_tags(self, tags_list):
        """Définir les tags à partir d'une liste"""
        self.tags = ', '.join(tags_list)
    
    @property
    def owner(self):
        """Propriétaire de la visualisation"""
        return self.analysis.owner
    
    @property
    def dataset(self):
        """Dataset source de la visualisation"""
        return self.analysis.dataset
    
    def can_view(self, user):
        """Vérifier si l'utilisateur peut voir cette visualisation"""
        return self.is_public or self.analysis.owner == user
    
    def can_edit(self, user):
        """Vérifier si l'utilisateur peut modifier cette visualisation"""
        return self.analysis.owner == user