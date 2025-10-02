from django.db import models
from django.contrib.auth import get_user_model
import json
import uuid
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
    


class ChatSession(models.Model):
    """Session de chat pour l'analyse de données"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200, default="Nouvelle conversation")
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    
    # Dataset associé à la session
    dataset = models.ForeignKey('Dataset', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Métadonnées de la session
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Configuration de l'IA
    model_used = models.CharField(max_length=50, default='gpt-4-turbo-preview')
    system_prompt = models.TextField(default="Vous êtes un assistant expert en analyse de données.")
    
    # Statistiques
    message_count = models.PositiveIntegerField(default=0)
    analysis_count = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Session de Chat"
        verbose_name_plural = "Sessions de Chat"
    
    def __str__(self):
        return f"{self.title} - {self.owner.username}"
    
    def get_last_messages(self, count=10):
        """Récupérer les derniers messages de la conversation"""
        return self.messages.order_by('-created_at')[:count]
    
    def update_title_from_first_message(self):
        """Mettre à jour le titre basé sur le premier message"""
        first_message = self.messages.filter(role='user').first()
        if first_message and self.title == "Nouvelle conversation":
            # Prendre les 50 premiers caractères du message
            new_title = first_message.content[:50]
            if len(first_message.content) > 50:
                new_title += "..."
            self.title = new_title
            self.save(update_fields=['title'])

class ChatMessage(models.Model):
    """Message dans une session de chat"""
    ROLE_CHOICES = [
        ('user', 'Utilisateur'),
        ('assistant', 'Assistant'),
        ('system', 'Système'),
        ('function', 'Fonction'),
    ]
    
    MESSAGE_TYPES = [
        ('text', 'Texte'),
        ('analysis', 'Analyse'),
        ('visualization', 'Visualisation'),
        ('code', 'Code'),
        ('error', 'Erreur'),
        ('data_summary', 'Résumé de données'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    
    # Contenu du message
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPES, default='text')
    
    # Métadonnées
    created_at = models.DateTimeField(auto_now_add=True)
    tokens_used = models.PositiveIntegerField(default=0)
    
    # Données structurées (pour les analyses, visualisations, etc.)
    structured_data = models.JSONField(default=dict, blank=True)
    
    # Relations avec d'autres objets
    related_analysis = models.ForeignKey('Analysis', on_delete=models.SET_NULL, null=True, blank=True)
    related_visualization = models.ForeignKey('Visualization', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Feedback utilisateur
    is_helpful = models.BooleanField(null=True, blank=True)
    user_feedback = models.TextField(blank=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = "Message de Chat"
        verbose_name_plural = "Messages de Chat"
    
    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:50]}..."
    
    def is_from_user(self):
        return self.role == 'user'
    
    def is_from_assistant(self):
        return self.role == 'assistant'

class DataAnalysisQuery(models.Model):
    """Requête d'analyse de données générée par l'IA"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='analysis_queries')
    
    # Query details
    query_text = models.TextField()  # Question originale de l'utilisateur
    analysis_type = models.CharField(max_length=50)  # Type d'analyse déterminé par l'IA
    
    # Code Python généré
    python_code = models.TextField()
    
    # Résultats
    execution_status = models.CharField(max_length=20, default='pending')  # pending, success, error
    results = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    
    # Métadonnées
    created_at = models.DateTimeField(auto_now_add=True)
    execution_time = models.FloatField(null=True, blank=True)  # Temps d'exécution en secondes
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."

class ChatAnalyticsPreset(models.Model):
    """Presets d'analyses prédéfinies pour le chat"""
    name = models.CharField(max_length=100)
    description = models.TextField()
    prompt_template = models.TextField()
    
    # Catégorie
    category = models.CharField(max_length=50, choices=[
        ('descriptive', 'Statistiques descriptives'),
        ('correlation', 'Analyse de corrélation'),
        ('regression', 'Régression'),
        ('classification', 'Classification'),
        ('clustering', 'Clustering'),
        ('timeseries', 'Séries temporelles'),
        ('visualization', 'Visualisation'),
        ('custom', 'Personnalisé'),
    ])
    
    # Configuration
    requires_target_column = models.BooleanField(default=False)
    requires_feature_columns = models.BooleanField(default=False)
    min_numeric_columns = models.PositiveIntegerField(default=0)
    min_categorical_columns = models.PositiveIntegerField(default=0)
    
    # Métadonnées
    is_active = models.BooleanField(default=True)
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['category', 'name']
    
    def __str__(self):
        return f"{self.get_category_display()}: {self.name}"