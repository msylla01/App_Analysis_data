from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, DetailView, CreateView, DeleteView, UpdateView
from django.contrib import messages
from django.urls import reverse_lazy
from django.db.models import Count, Q, Sum
from django.utils.safestring import mark_safe
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from datetime import timedelta
from .models import Dataset, Analysis, Visualization
from .forms import AnalysisForm
from .services import DataAnalysisService
import json

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'analytics/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Statistiques générales
        context['datasets_count'] = Dataset.objects.filter(owner=user).count()
        context['analyses_count'] = Analysis.objects.filter(owner=user).count()
        context['visualizations_count'] = Visualization.objects.filter(analysis__owner=user).count()
        
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
        
        # Activité des 7 derniers jours
        seven_days_ago = timezone.now() - timedelta(days=7)
        context['recent_activity'] = {
            'datasets': Dataset.objects.filter(owner=user, created_at__gte=seven_days_ago).count(),
            'analyses': Analysis.objects.filter(owner=user, created_at__gte=seven_days_ago).count(),
            'visualizations': Visualization.objects.filter(
                analysis__owner=user, created_at__gte=seven_days_ago
            ).count(),
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
                    'id': visualization.dataset.id,
                    'name': visualization.dataset.name,
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