from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView, TemplateView, UpdateView
from django.contrib.auth import login
from django.contrib import messages
from django.urls import reverse_lazy
from django.utils import timezone
from datetime import timedelta
from .models import CustomUser, UserProfile
from .forms import CustomUserCreationForm, UserProfileForm
from analytics.models import Dataset, Analysis

class CustomLoginView(LoginView):
    template_name = 'users/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('analytics:dashboard')
    
    def form_valid(self, form):
        messages.success(self.request, f'Bienvenue {form.get_user().username} !')
        return super().form_valid(form)

class CustomLogoutView(LogoutView):
    next_page = reverse_lazy('users:login')
    
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            messages.info(request, 'Vous avez été déconnecté avec succès.')
        return super().dispatch(request, *args, **kwargs)

class RegisterView(CreateView):
    model = CustomUser
    form_class = CustomUserCreationForm
    template_name = 'users/register.html'
    success_url = reverse_lazy('analytics:dashboard')
    
    def form_valid(self, form):
        response = super().form_valid(form)
        # Créer le profil utilisateur
        UserProfile.objects.create(user=self.object)
        # Connecter automatiquement l'utilisateur
        login(self.request, self.object)
        messages.success(self.request, 'Compte créé avec succès !')
        return response

class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = 'users/profile.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Créer le profil s'il n'existe pas
        profile, created = UserProfile.objects.get_or_create(user=user)
        context['profile'] = profile
        
        # Statistiques de l'utilisateur
        context['datasets_count'] = Dataset.objects.filter(owner=user).count()
        context['analyses_count'] = Analysis.objects.filter(owner=user).count()
        
        # Calculer les jours d'activité
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_analyses = Analysis.objects.filter(
            owner=user, 
            created_at__gte=thirty_days_ago
        ).dates('created_at', 'day')
        context['days_active'] = len(recent_analyses)
        
        # Activité récente (dernières analyses et datasets)
        recent_activities = []
        
        # Récentes analyses
        recent_analyses_list = Analysis.objects.filter(owner=user).order_by('-created_at')[:5]
        for analysis in recent_analyses_list:
            recent_activities.append({
                'title': f'Analyse: {analysis.title}',
                'description': f'Type: {analysis.get_analysis_type_display()} - {analysis.get_status_display()}',
                'created_at': analysis.created_at,
                'type': 'analysis'
            })
        
        # Récents datasets
        recent_datasets = Dataset.objects.filter(owner=user).order_by('-created_at')[:3]
        for dataset in recent_datasets:
            recent_activities.append({
                'title': f'Dataset: {dataset.name}',
                'description': f'Type: {dataset.get_dataset_type_display()}',
                'created_at': dataset.created_at,
                'type': 'dataset'
            })
        
        # Trier par date de création (plus récent en premier)
        recent_activities.sort(key=lambda x: x['created_at'], reverse=True)
        context['recent_activities'] = recent_activities[:8]  # Garder les 8 plus récents
        
        return context

class EditProfileView(LoginRequiredMixin, UpdateView):
    model = UserProfile
    form_class = UserProfileForm
    template_name = 'users/edit_profile.html'
    success_url = reverse_lazy('users:profile')
    
    def get_object(self):
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile
    
    def form_valid(self, form):
        # Mettre à jour aussi les informations de l'utilisateur
        user = self.request.user
        
        # Récupérer les données du formulaire pour l'utilisateur
        user.first_name = self.request.POST.get('first_name', '')
        user.last_name = self.request.POST.get('last_name', '')
        user.email = self.request.POST.get('email', user.email)
        user.username = self.request.POST.get('username', user.username)
        user.company = self.request.POST.get('company', '')
        user.role = self.request.POST.get('role', '')
        
        try:
            user.save()
            messages.success(self.request, 'Profil mis à jour avec succès !')
        except Exception as e:
            messages.error(self.request, f'Erreur lors de la mise à jour: {str(e)}')
            return self.form_invalid(form)
        
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        return context