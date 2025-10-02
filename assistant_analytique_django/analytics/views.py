from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, DetailView, CreateView, DeleteView, UpdateView
from django.contrib import messages
from django.urls import reverse_lazy
from django.db.models import Count, Q, Sum, Max
from django.utils.safestring import mark_safe
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from datetime import timedelta
from .models import Dataset, Analysis, Visualization, ChatSession, ChatMessage, DataAnalysisQuery
from .forms import AnalysisForm
from .services import DataAnalysisService
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

# ===== VUES EXISTANTES (Dashboard, Datasets, Analyses, Visualisations) =====

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'analytics/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Statistiques générales existantes
        context['datasets_count'] = Dataset.objects.filter(owner=user).count()
        context['analyses_count'] = Analysis.objects.filter(owner=user).count()
        context['visualizations_count'] = Visualization.objects.filter(analysis__owner=user).count()
        
        # NOUVELLES STATS CHAT
        context['chat_sessions_count'] = ChatSession.objects.filter(owner=user).count()
        context['chat_messages_count'] = ChatMessage.objects.filter(session__owner=user).count()
        
        # Analyses par statut
        context['completed_analyses'] = Analysis.objects.filter(owner=user, status='completed').count()
        context['running_analyses'] = Analysis.objects.filter(owner=user, status='running').count()
        context['failed_analyses'] = Analysis.objects.filter(owner=user, status='failed').count()
        
        # Datasets récents
        context['recent_datasets'] = Dataset.objects.filter(owner=user).order_by('-created_at')[:5]
        
        # Analyses récentes
        context['recent_analyses'] = Analysis.objects.filter(owner=user).order_by('-created_at')[:5]
        
        # Visualisations récentes
        context['recent_visualizations'] = Visualization.objects.filter(
            analysis__owner=user
        ).order_by('-created_at')[:6]
        
        # CONVERSATIONS RÉCENTES
        context['recent_chat_sessions'] = ChatSession.objects.filter(
            owner=user
        ).order_by('-updated_at')[:3]
        
        # Activité des 7 derniers jours
        seven_days_ago = timezone.now() - timedelta(days=7)
        context['recent_activity'] = {
            'datasets': Dataset.objects.filter(owner=user, created_at__gte=seven_days_ago).count(),
            'analyses': Analysis.objects.filter(owner=user, created_at__gte=seven_days_ago).count(),
            'visualizations': Visualization.objects.filter(
                analysis__owner=user, created_at__gte=seven_days_ago
            ).count(),
            'chat_sessions': ChatSession.objects.filter(owner=user, created_at__gte=seven_days_ago).count(),
        }
        
        return context

class DatasetListView(LoginRequiredMixin, ListView):
    model = Dataset
    template_name = 'analytics/datasets.html'
    context_object_name = 'datasets'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = Dataset.objects.filter(owner=self.request.user).order_by('-created_at')
        
        # Recherche
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(name__icontains=search)
        
        return queryset

class DatasetDetailView(LoginRequiredMixin, DetailView):
    model = Dataset
    template_name = 'analytics/dataset_detail.html'
    context_object_name = 'dataset'
    
    def get_queryset(self):
        return Dataset.objects.filter(owner=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.get_object()
        
        # Analyses liées
        context['analyses'] = Analysis.objects.filter(dataset=dataset).order_by('-created_at')
        
        # Aperçu du dataset
        try:
            service = DataAnalysisService()
            context['preview_data'] = service.get_dataset_preview(dataset)
        except Exception as e:
            messages.error(self.request, f'Erreur lors du chargement de l\'aperçu: {str(e)}')
            context['preview_data'] = None
        
        return context

class CreateAnalysisView(LoginRequiredMixin, CreateView):
    model = Analysis
    form_class = AnalysisForm
    template_name = 'analytics/create_analysis.html'
    
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        dataset_id = self.kwargs.get('pk')
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        kwargs['dataset'] = dataset
        return kwargs
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset_id = self.kwargs.get('pk')
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        context['dataset'] = dataset
        
        # Ajouter des informations sur la compatibilité du dataset
        try:
            service = DataAnalysisService()
            import pandas as pd
            df = service.load_dataset(dataset)
            
            # Vérifier la compatibilité pour chaque type d'analyse
            compatibility = {}
            for analysis_type in ['descriptive', 'correlation', 'regression', 'clustering']:
                validation = service.validate_dataset_for_analysis(df, analysis_type)
                compatibility[analysis_type] = validation
            
            context['compatibility'] = compatibility
            
            # Informations sur les colonnes
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            date_columns = []
            
            # Détecter les colonnes de date potentielles
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].dropna().head(100))
                        date_columns.append(col)
                    except:
                        pass
            
            context['numeric_columns'] = numeric_columns
            context['categorical_columns'] = categorical_columns
            context['date_columns'] = date_columns
            context['total_missing'] = df.isnull().sum().sum()
            context['missing_percentage'] = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            
        except Exception as e:
            messages.warning(self.request, f'Impossible d\'analyser la compatibilité du dataset: {str(e)}')
            context['compatibility'] = {}
        
        return context
    
    def form_valid(self, form):
        dataset_id = self.kwargs.get('pk')
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        
        form.instance.dataset = dataset
        form.instance.owner = self.request.user
        
        # Obtenir les paramètres du formulaire
        parameters = form.get_parameters()
        form.instance.parameters = parameters
        
        # Valider avant de créer
        try:
            service = DataAnalysisService()
            df = service.load_dataset(dataset)
            validation = service.validate_dataset_for_analysis(df, form.instance.analysis_type)
            
            if not validation['is_valid']:
                for error in validation['errors']:
                    messages.error(self.request, error)
                return self.form_invalid(form)
            
            # Afficher les avertissements
            for warning in validation.get('warnings', []):
                messages.warning(self.request, warning)
                
        except Exception as e:
            messages.error(self.request, f'Erreur de validation: {str(e)}')
            return self.form_invalid(form)
        
        response = super().form_valid(form)
        
        # Lancer l'analyse
        try:
            service = DataAnalysisService()
            service.run_analysis(self.object)
            messages.success(self.request, 'Analyse créée et exécutée avec succès !')
        except Exception as e:
            messages.error(self.request, f'Erreur lors de l\'exécution de l\'analyse: {str(e)}')
        
        return response
    
    def get_success_url(self):
        return reverse_lazy('analytics:analysis_detail', kwargs={'pk': self.object.pk})

class AnalysisListView(LoginRequiredMixin, ListView):
    model = Analysis
    template_name = 'analytics/analyses.html'
    context_object_name = 'analyses'
    paginate_by = 12
    
    def get_queryset(self):
        return Analysis.objects.filter(owner=self.request.user).order_by('-created_at')

class AnalysisDetailView(LoginRequiredMixin, DetailView):
    model = Analysis
    template_name = 'analytics/analysis_detail.html'
    context_object_name = 'analysis'
    
    def get_queryset(self):
        return Analysis.objects.filter(owner=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        analysis = self.get_object()
        
        # Préparer les données de résultats pour l'affichage
        if analysis.results:
            try:
                context['results_json'] = json.dumps(analysis.results, indent=2, ensure_ascii=False)
            except Exception as e:
                context['results_json'] = f"Erreur lors du formatage: {str(e)}"
        
        # Visualisations liées avec données JSON sécurisées
        visualizations = Visualization.objects.filter(analysis=analysis).order_by('-created_at')
        viz_list = []
        
        for viz in visualizations:
            try:
                viz_data = {
                    'id': viz.id,
                    'title': viz.title,
                    'description': viz.description,
                    'chart_type': viz.chart_type,
                    'created_at': viz.created_at,
                    'chart_data_json': json.dumps(viz.chart_data) if viz.chart_data else '{}',
                    'chart_config_json': json.dumps(viz.chart_config) if viz.chart_config else '{}',
                    'has_data': bool(viz.chart_data),
                    'has_config': bool(viz.chart_config),
                }
                viz_list.append(viz_data)
            except Exception as e:
                print(f"Erreur lors de la préparation de la visualisation {viz.id}: {str(e)}")
                continue
        
        context['visualizations'] = visualizations
        context['visualizations_data'] = viz_list
        
        return context

class AnalysisDeleteView(LoginRequiredMixin, DeleteView):
    model = Analysis
    template_name = 'analytics/analysis_confirm_delete.html'
    success_url = reverse_lazy('analytics:analyses')
    
    def get_queryset(self):
        return Analysis.objects.filter(owner=self.request.user)

# Vues pour les visualisations
class VisualizationListView(LoginRequiredMixin, ListView):
    model = Visualization
    template_name = 'analytics/visualizations.html'
    context_object_name = 'visualizations'
    paginate_by = 15
    
    def get_queryset(self):
        queryset = Visualization.objects.filter(
            analysis__owner=self.request.user
        ).select_related('analysis', 'analysis__dataset').order_by('-created_at')
        
        # Filtres
        chart_type = self.request.GET.get('type')
        if chart_type:
            queryset = queryset.filter(chart_type=chart_type)
        
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) |
                Q(description__icontains=search) |
                Q(tags__icontains=search)
            )
        
        analysis_id = self.request.GET.get('analysis')
        if analysis_id:
            queryset = queryset.filter(analysis_id=analysis_id)
        
        # Tri
        sort = self.request.GET.get('sort', '-created_at')
        if sort in ['created_at', '-created_at', 'title', '-title', 'views_count', '-views_count']:
            queryset = queryset.order_by(sort)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Statistiques
        user_visualizations = Visualization.objects.filter(analysis__owner=self.request.user)
        context['stats'] = {
            'total_visualizations': user_visualizations.count(),
            'total_views': user_visualizations.aggregate(total=Sum('views_count'))['total'] or 0,
            'total_downloads': user_visualizations.aggregate(total=Sum('downloads_count'))['total'] or 0,
            'chart_types': user_visualizations.values('chart_type').annotate(count=Count('id')),
        }
        
        # Types de graphiques disponibles
        context['chart_types'] = Visualization.CHART_TYPES
        
        # Analyses disponibles pour le filtre
        context['analyses'] = Analysis.objects.filter(owner=self.request.user).order_by('-created_at')
        
        # Paramètres de filtre actuels
        context['current_filters'] = {
            'type': self.request.GET.get('type', ''),
            'search': self.request.GET.get('search', ''),
            'analysis': self.request.GET.get('analysis', ''),
            'sort': self.request.GET.get('sort', '-created_at'),
        }
        
        return context

class VisualizationDetailView(LoginRequiredMixin, DetailView):
    model = Visualization
    template_name = 'analytics/visualization_detail.html'
    context_object_name = 'visualization'
    
    def get_queryset(self):
        return Visualization.objects.filter(
            analysis__owner=self.request.user
        ).select_related('analysis', 'analysis__dataset')
    
    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        # Incrémenter le nombre de vues
        obj.increment_views()
        return obj
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        visualization = self.get_object()
        
        # Préparer les données JSON de manière sécurisée
        try:
            if visualization.chart_data:
                chart_data_str = json.dumps(visualization.chart_data, ensure_ascii=False)
                context['chart_data_json'] = mark_safe(chart_data_str)
            else:
                context['chart_data_json'] = mark_safe('{}')
                
            if visualization.chart_config:
                chart_config_str = json.dumps(visualization.chart_config, ensure_ascii=False)
                context['chart_config_json'] = mark_safe(chart_config_str)
            else:
                context['chart_config_json'] = mark_safe('{}')
                
        except Exception as e:
            context['chart_data_json'] = mark_safe('{}')
            context['chart_config_json'] = mark_safe('{}')
            messages.error(self.request, f'Erreur lors du chargement des données: {str(e)}')
        
        # Visualisations similaires
        similar_visualizations = Visualization.objects.filter(
            analysis__owner=self.request.user,
            chart_type=visualization.chart_type
        ).exclude(id=visualization.id)[:4]
        context['similar_visualizations'] = similar_visualizations
        
        # Autres visualisations de la même analyse
        related_visualizations = Visualization.objects.filter(
            analysis=visualization.analysis
        ).exclude(id=visualization.id)
        context['related_visualizations'] = related_visualizations
        
        return context

class VisualizationCreateView(LoginRequiredMixin, TemplateView):
    template_name = 'analytics/visualization_create.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Analyses disponibles
        context['analyses'] = Analysis.objects.filter(
            owner=self.request.user,
            status='completed'
        ).order_by('-created_at')
        
        # Types de graphiques
        context['chart_types'] = Visualization.CHART_TYPES
        
        return context

class VisualizationUpdateView(LoginRequiredMixin, UpdateView):
    model = Visualization
    template_name = 'analytics/visualization_update.html'
    fields = ['title', 'description', 'tags', 'status', 'is_public']
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get_success_url(self):
        return reverse_lazy('analytics:visualization_detail', kwargs={'pk': self.object.pk})

class VisualizationDeleteView(LoginRequiredMixin, DeleteView):
    model = Visualization
    template_name = 'analytics/visualization_confirm_delete.html'
    success_url = reverse_lazy('analytics:visualizations')
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)

# Vues API pour AJAX
class VisualizationImageView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        visualization.increment_downloads()
        
        return JsonResponse({
            'message': 'Use Chart.js toDataURL() method on the frontend',
            'visualization_id': visualization.id
        })

class VisualizationStatsView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        
        stats = {
            'views': visualization.views_count,
            'downloads': visualization.downloads_count,
            'created': visualization.created_at.isoformat(),
            'updated': visualization.updated_at.isoformat(),
            'chart_type': visualization.get_chart_type_display(),
            'data_points': len(visualization.chart_data.get('labels', [])),
            'datasets': len(visualization.chart_data.get('datasets', [])),
        }
        
        return JsonResponse(stats)

class VisualizationExportView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        format_type = request.GET.get('format', 'json')
        
        if format_type == 'json':
            data = {
                'id': visualization.id,
                'title': visualization.title,
                'description': visualization.description,
                'chart_type': visualization.chart_type,
                'chart_data': visualization.chart_data,
                'chart_config': visualization.chart_config,
                'created_at': visualization.created_at.isoformat(),
                'analysis': {
                    'id': visualization.analysis.id,
                    'title': visualization.analysis.title,
                    'type': visualization.analysis.analysis_type,
                },
                'dataset': {
                    'id': visualization.analysis.dataset.id,
                    'name': visualization.analysis.dataset.name,
                }
            }
            
            response = HttpResponse(
                json.dumps(data, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            response['Content-Disposition'] = f'attachment; filename="{visualization.title}.json"'
            
            visualization.increment_downloads()
            return response
        
        return JsonResponse({'error': 'Format not supported'}, status=400)

class PublicVisualizationDetailView(DetailView):
    model = Visualization
    template_name = 'analytics/visualization_public.html'
    context_object_name = 'visualization'
    
    def get_queryset(self):
        return Visualization.objects.filter(is_public=True, status='published')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        visualization = self.get_object()
        
        # Incrémenter les vues pour les visualisations publiques aussi
        visualization.increment_views()
        
        # Préparer les données JSON
        try:
            context['chart_data_json'] = mark_safe(json.dumps(visualization.chart_data, ensure_ascii=False))
            context['chart_config_json'] = mark_safe(json.dumps(visualization.chart_config, ensure_ascii=False))
        except Exception:
            context['chart_data_json'] = mark_safe('{}')
            context['chart_config_json'] = mark_safe('{}')
        
        return context

# ===== NOUVELLES VUES POUR LE CHAT =====

class ChatSessionListView(LoginRequiredMixin, ListView):
    """Liste des sessions de chat de l'utilisateur"""
    model = ChatSession
    template_name = 'analytics/chat/chat_sessions.html'
    context_object_name = 'sessions'
    paginate_by = 20
    
    def get_queryset(self):
        return ChatSession.objects.filter(
            owner=self.request.user,
            is_active=True
        ).annotate(
            last_message_time=Max('messages__created_at'),
            message_count_actual=Count('messages')
        ).order_by('-updated_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Statistiques de l'utilisateur
        user_sessions = ChatSession.objects.filter(owner=self.request.user)
        context['stats'] = {
            'total_sessions': user_sessions.count(),
            'active_sessions': user_sessions.filter(is_active=True).count(),
            'total_messages': ChatMessage.objects.filter(session__owner=self.request.user).count(),
            'total_analyses': DataAnalysisQuery.objects.filter(
                message__session__owner=self.request.user
            ).count(),
        }
        
        # Datasets disponibles pour créer une nouvelle session
        context['available_datasets'] = Dataset.objects.filter(
            owner=self.request.user
        ).order_by('-created_at')[:10]
        
        return context

# Mise à jour de ChatSessionDetailView pour des suggestions dynamiques

class ChatSessionDetailView(LoginRequiredMixin, DetailView):
    """Interface de chat principale avec IA"""
    model = ChatSession
    template_name = 'analytics/chat/chat_interface.html'
    context_object_name = 'session'
    
    def get_queryset(self):
        return ChatSession.objects.filter(owner=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        session = self.get_object()
        
        # Messages de la session
        messages = session.messages.order_by('created_at')
        context['messages'] = messages
        
        # Datasets disponibles
        context['available_datasets'] = Dataset.objects.filter(
            owner=self.request.user
        ).order_by('-created_at')
        
        # Suggestions d'analyse dynamiques basées sur l'IA
        try:
            from .ai_service import DataAnalysisAI
            ai_service = DataAnalysisAI()
            context['analysis_suggestions'] = ai_service.get_analysis_suggestions(session.dataset)
        except Exception as e:
            logger.warning(f"Erreur suggestions IA: {e}")
            # Suggestions de fallback
            if session.dataset:
                context['analysis_suggestions'] = [
                    "Montre-moi un résumé de mes données",
                    "Quelles sont les corrélations entre les variables ?",
                    "Y a-t-il des valeurs aberrantes ?",
                    "Crée un graphique des principales variables",
                    "Analyse la distribution des données",
                    "Calcule les statistiques descriptives",
                    "Fais une analyse de corrélation",
                    "Détecte les patterns dans les données"
                ]
            else:
                context['analysis_suggestions'] = [
                    "Sélectionnez d'abord un dataset pour commencer l'analyse",
                    "Uploadez un nouveau fichier de données",
                    "Explorez vos datasets existants",
                    "Demandez de l'aide sur l'analyse de données"
                ]
        
        # Informations sur le dataset avec gestion d'erreur
        if session.dataset:
            try:
                from .services import DataAnalysisService
                service = DataAnalysisService()
                df = service.load_dataset(session.dataset)
                context['dataset_info'] = {
                    'name': session.dataset.name,
                    'shape': df.shape,
                    'columns': df.columns.tolist()[:10],
                    'total_columns': len(df.columns),
                    'dtypes': df.dtypes.value_counts().to_dict(),
                    'missing_values': df.isnull().sum().sum(),
                    'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
                }
            except Exception as e:
                logger.error(f"Erreur info dataset: {e}")
                context['dataset_info'] = {
                    'name': session.dataset.name,
                    'error': str(e)
                }
        
        return context

class ChatSessionCreateView(LoginRequiredMixin, CreateView):
    """Créer une nouvelle session de chat"""
    model = ChatSession
    fields = ['title']
    template_name = 'analytics/chat/create_session.html'
    
    def form_valid(self, form):
        form.instance.owner = self.request.user
        
        # Associer un dataset si fourni
        dataset_id = self.request.POST.get('dataset')
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id, owner=self.request.user)
                form.instance.dataset = dataset
                
                # Générer un titre automatique si non fourni
                if not form.instance.title or form.instance.title == "Nouvelle conversation":
                    form.instance.title = f"Analyse de {dataset.name}"
            except Dataset.DoesNotExist:
                messages.warning(self.request, "Dataset non trouvé")
        
        response = super().form_valid(form)
        
        # Créer un message de bienvenue
        welcome_message = self._create_welcome_message()
        ChatMessage.objects.create(
            session=self.object,
            role='assistant',
            content=welcome_message,
            message_type='text'
        )
        
        return response
    
    def _create_welcome_message(self):
        """Créer un message de bienvenue personnalisé"""
        if self.object.dataset:
            return f"""Bonjour ! 👋

Je suis votre assistant d'analyse de données. Je vais vous aider à explorer et analyser le dataset **{self.object.dataset.name}**.

Vous pouvez me poser des questions en langage naturel comme :
• "Montre-moi un résumé des données"
• "Quelles sont les corrélations entre les variables ?"
• "Crée un graphique des ventes par mois"
• "Y a-t-il des valeurs aberrantes ?"

Comment puis-je vous aider aujourd'hui ?"""
        else:
            return """Bonjour ! 👋

Je suis votre assistant d'analyse de données. Pour commencer, sélectionnez un dataset à analyser ou posez-moi des questions générales sur l'analyse de données.

Comment puis-je vous aider aujourd'hui ?"""
    
    def get_success_url(self):
        return reverse_lazy('analytics:chat_session_detail', kwargs={'pk': self.object.pk})
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['available_datasets'] = Dataset.objects.filter(
            owner=self.request.user
        ).order_by('-created_at')
        return context

# Remplacez la classe ChatMessageCreateView existante par cette version complète

import asyncio
from .ai_service import DataAnalysisAI

@method_decorator(csrf_exempt, name='dispatch')
class ChatMessageCreateView(LoginRequiredMixin, TemplateView):
    """API pour envoyer des messages dans le chat avec IA OpenAI"""
    
    def post(self, request, session_id):
        try:
            # Récupérer la session
            session = get_object_or_404(
                ChatSession, 
                id=session_id, 
                owner=request.user
            )
            
            # Récupérer le message
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({
                    'success': False,
                    'error': 'Message vide'
                }, status=400)
            
            logger.info(f"Processing message from user {request.user.username}: {user_message[:100]}")
            
            try:
                # Utiliser le service IA pour traiter le message
                ai_service = DataAnalysisAI()
                
                # Créer un nouveau event loop pour la fonction async
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Traiter le message avec l'IA
                    response_message = loop.run_until_complete(
                        ai_service.process_user_message(session, user_message)
                    )
                    
                    # Récupérer le message utilisateur créé par le service
                    user_msg = session.messages.filter(role='user').last()
                    
                    # Préparer la réponse JSON
                    response_data = {
                        'success': True,
                        'user_message': {
                            'id': str(user_msg.id),
                            'content': user_message,
                            'timestamp': user_msg.created_at.isoformat(),
                            'role': 'user'
                        },
                        'assistant_message': {
                            'id': str(response_message.id),
                            'content': response_message.content,
                            'timestamp': response_message.created_at.isoformat(),
                            'role': 'assistant',
                            'message_type': response_message.message_type,
                            'structured_data': response_message.structured_data
                        }
                    }
                    
                    logger.info(f"Message processed successfully: {response_message.message_type}")
                    return JsonResponse(response_data)
                    
                finally:
                    loop.close()
                
            except Exception as ai_error:
                logger.error(f"Erreur AI service: {str(ai_error)}")
                import traceback
                traceback.print_exc()
                
                # Créer un message utilisateur manual si le service IA échoue
                user_msg = ChatMessage.objects.create(
                    session=session,
                    role='user',
                    content=user_message,
                    message_type='text'
                )
                
                # Créer une réponse d'erreur de fallback
                error_response = f"""❌ **Désolé, une erreur s'est produite avec l'IA.**

**Erreur:** {str(ai_error)}

Je suis temporairement indisponible pour l'analyse avancée. Voici ce que vous pouvez essayer :

🔧 **Solutions possibles :**
• Vérifiez votre clé API OpenAI
• Réessayez votre demande  
• Contactez l'administrateur si le problème persiste

💡 **En attendant, voici quelques suggestions :**
• Explorez vos données avec des outils classiques
• Consultez la documentation sur l'analyse de données
• Préparez vos questions pour quand l'IA sera de nouveau disponible

Je suis désolé pour ce désagrément ! 😔"""
                
                error_message = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=error_response,
                    message_type='error',
                    structured_data={'error': str(ai_error), 'fallback': True}
                )
                
                # Mettre à jour les compteurs de session
                session.message_count += 2
                session.save()
                
                return JsonResponse({
                    'success': True,
                    'user_message': {
                        'id': str(user_msg.id),
                        'content': user_message,
                        'timestamp': user_msg.created_at.isoformat(),
                        'role': 'user'
                    },
                    'assistant_message': {
                        'id': str(error_message.id),
                        'content': error_response,
                        'timestamp': error_message.created_at.isoformat(),
                        'role': 'assistant',
                        'message_type': 'error'
                    }
                })
            
        except Exception as e:
            logger.error(f"Erreur dans ChatMessageCreateView: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return JsonResponse({
                'success': False,
                'error': f'Erreur serveur: {str(e)}'
            }, status=500)

class ChatSessionUpdateDatasetView(LoginRequiredMixin, TemplateView):
    """Changer le dataset d'une session"""
    
    def post(self, request, session_id):
        try:
            session = get_object_or_404(
                ChatSession, 
                id=session_id, 
                owner=request.user
            )
            
            data = json.loads(request.body)
            dataset_id = data.get('dataset_id')
            
            if dataset_id:
                dataset = get_object_or_404(
                    Dataset, 
                    id=dataset_id, 
                    owner=request.user
                )
                session.dataset = dataset
                session.save()
                
                # Créer un message système pour notifier le changement
                ChatMessage.objects.create(
                    session=session,
                    role='system',
                    content=f"Dataset changé pour: **{dataset.name}**",
                    message_type='text'
                )
                
                return JsonResponse({
                    'success': True,
                    'dataset': {
                        'id': dataset.id,
                        'name': dataset.name
                    }
                })
            else:
                # Supprimer le dataset de la session
                session.dataset = None
                session.save()
                
                return JsonResponse({
                    'success': True,
                    'dataset': None
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

class ChatSessionDeleteView(LoginRequiredMixin, DeleteView):
    """Supprimer une session de chat"""
    model = ChatSession
    template_name = 'analytics/chat/confirm_delete.html'
    success_url = reverse_lazy('analytics:chat_sessions')
    
    def get_queryset(self):
        return ChatSession.objects.filter(owner=self.request.user)

@require_http_methods(["POST"])
def chat_message_feedback(request, message_id):
    """Enregistrer le feedback sur un message"""
    try:
        message = get_object_or_404(
            ChatMessage, 
            id=message_id, 
            session__owner=request.user
        )
        
        data = json.loads(request.body)
        is_helpful = data.get('is_helpful')
        feedback_text = data.get('feedback', '')
        
        message.is_helpful = is_helpful
        message.user_feedback = feedback_text
        message.save()
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

class ChatExportView(LoginRequiredMixin, TemplateView):
    """Exporter une conversation"""
    
    def get(self, request, session_id):
        try:
            session = get_object_or_404(
                ChatSession, 
                id=session_id, 
                owner=request.user
            )
            
            format_type = request.GET.get('format', 'json')
            
            # Préparer les données
            export_data = {
                'session': {
                    'id': str(session.id),
                    'title': session.title,
                    'created_at': session.created_at.isoformat(),
                    'dataset': session.dataset.name if session.dataset else None,
                },
                'messages': []
            }
            
            for message in session.messages.order_by('created_at'):
                export_data['messages'].append({
                    'id': str(message.id),
                    'role': message.role,
                    'content': message.content,
                    'message_type': message.message_type,
                    'created_at': message.created_at.isoformat(),
                })
            
            if format_type == 'json':
                response = HttpResponse(
                    json.dumps(export_data, indent=2, ensure_ascii=False),
                    content_type='application/json'
                )
                response['Content-Disposition'] = f'attachment; filename="chat_{session.id}.json"'
                return response
            
            elif format_type == 'txt':
                # Format texte simple
                text_content = f"Conversation: {session.title}\n"
                text_content += f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                if session.dataset:
                    text_content += f"Dataset: {session.dataset.name}\n"
                text_content += "\n" + "="*50 + "\n\n"
                
                for message in session.messages.order_by('created_at'):
                    text_content += f"[{message.created_at.strftime('%H:%M:%S')}] "
                    text_content += f"{message.get_role_display()}:\n"
                    text_content += f"{message.content}\n\n"
                
                response = HttpResponse(text_content, content_type='text/plain')
                response['Content-Disposition'] = f'attachment; filename="chat_{session.id}.txt"'
                return response
            
            else:
                return JsonResponse({'error': 'Format non supporté'}, status=400)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

class ChatAnalyticsView(LoginRequiredMixin, TemplateView):
    """Analytics et insights sur l'utilisation du chat"""
    template_name = 'analytics/chat/analytics.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Statistiques générales
        user_sessions = ChatSession.objects.filter(owner=user)
        user_messages = ChatMessage.objects.filter(session__owner=user)
        
        context['stats'] = {
            'total_sessions': user_sessions.count(),
            'total_messages': user_messages.count(),
            'avg_messages_per_session': user_messages.count() / max(user_sessions.count(), 1),
            'active_sessions': user_sessions.filter(is_active=True).count(),
        }
        
        # Sessions récentes avec activité
        context['recent_sessions'] = user_sessions.annotate(
            last_activity=Max('messages__created_at'),
            total_messages=Count('messages')
        ).order_by('-last_activity')[:10]
        
        # Messages par jour (30 derniers jours)
        from django.db.models import Count
        from django.utils import timezone
        
        thirty_days_ago = timezone.now() - timedelta(days=30)
        daily_messages = user_messages.filter(
            created_at__gte=thirty_days_ago
        ).extra(
            select={'day': 'date(created_at)'}
        ).values('day').annotate(
            message_count=Count('id')
        ).order_by('day')
        
        context['daily_messages_data'] = list(daily_messages)
        
        # Distribution des types de messages
        message_types = user_messages.values('message_type').annotate(
            count=Count('id')
        )
        context['message_types_data'] = list(message_types)
        
        return context
    
# Ajoutez ces vues manquantes à la fin de votre fichier views.py

class VisualizationCreateView(LoginRequiredMixin, TemplateView):
    template_name = 'analytics/visualization_create.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['analyses'] = Analysis.objects.filter(
            owner=self.request.user,
            status='completed'
        ).order_by('-created_at')
        return context

class VisualizationUpdateView(LoginRequiredMixin, UpdateView):
    model = Visualization
    template_name = 'analytics/visualization_update.html'
    fields = ['title', 'description', 'tags', 'is_public']
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get_success_url(self):
        return reverse_lazy('analytics:visualization_detail', kwargs={'pk': self.object.pk})

class VisualizationDeleteView(LoginRequiredMixin, DeleteView):
    model = Visualization
    template_name = 'analytics/visualization_confirm_delete.html'
    success_url = reverse_lazy('analytics:visualizations')
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)

class VisualizationImageView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        visualization.increment_downloads()
        return JsonResponse({
            'message': 'Use Chart.js toDataURL() method on the frontend',
            'visualization_id': visualization.id
        })

class VisualizationStatsView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        stats = {
            'views': visualization.views_count,
            'downloads': visualization.downloads_count,
            'created': visualization.created_at.isoformat(),
            'updated': visualization.updated_at.isoformat(),
        }
        return JsonResponse(stats)

class VisualizationExportView(LoginRequiredMixin, DetailView):
    model = Visualization
    
    def get_queryset(self):
        return Visualization.objects.filter(analysis__owner=self.request.user)
    
    def get(self, request, *args, **kwargs):
        visualization = self.get_object()
        data = {
            'id': visualization.id,
            'title': visualization.title,
            'chart_data': visualization.chart_data,
            'chart_config': visualization.chart_config,
        }
        
        response = HttpResponse(
            json.dumps(data, indent=2, ensure_ascii=False),
            content_type='application/json'
        )
        response['Content-Disposition'] = f'attachment; filename="{visualization.title}.json"'
        visualization.increment_downloads()
        return response

class PublicVisualizationDetailView(DetailView):
    model = Visualization
    template_name = 'analytics/visualization_public.html'
    context_object_name = 'visualization'
    
    def get_queryset(self):
        return Visualization.objects.filter(is_public=True)