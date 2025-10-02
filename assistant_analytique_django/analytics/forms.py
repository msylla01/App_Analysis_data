from django import forms
from .models import Analysis
import json

class AnalysisForm(forms.ModelForm):
    # Champs supplémentaires pour les paramètres d'analyse
    target_column = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text="Colonne cible pour la régression"
    )
    
    feature_columns = forms.MultipleChoiceField(
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'form-select', 'size': '4'}),
        help_text="Colonnes explicatives pour la régression"
    )
    
    n_clusters = forms.IntegerField(
        min_value=2, 
        max_value=20, 
        initial=3,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Nombre de clusters"
    )
    
    date_column = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text="Colonne de date pour l'analyse temporelle"
    )
    
    value_column = forms.CharField(
        max_length=100, 
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text="Colonne de valeur pour l'analyse temporelle"
    )

    class Meta:
        model = Analysis
        fields = ['title', 'description', 'analysis_type']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Titre de l\'analyse'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Description de l\'analyse'
            }),
            'analysis_type': forms.Select(attrs={
                'class': 'form-select'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)
        
        if dataset:
            self.dataset = dataset
            # Charger les colonnes du dataset pour les choix
            try:
                from .services import DataAnalysisService
                service = DataAnalysisService()
                df = service.load_dataset(dataset)
                
                # Colonnes numériques pour régression et clustering
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                numeric_choices = [(col, col) for col in numeric_columns]
                
                # Toutes les colonnes pour date
                all_columns = df.columns.tolist()
                all_choices = [(col, col) for col in all_columns]
                
                # Mettre à jour les choix
                if numeric_choices:
                    self.fields['target_column'].widget.choices = [('', 'Sélectionner...')] + numeric_choices
                    self.fields['feature_columns'].choices = numeric_choices
                    self.fields['value_column'].widget.choices = [('', 'Sélectionner...')] + numeric_choices
                
                if all_choices:
                    self.fields['date_column'].widget.choices = [('', 'Sélectionner...')] + all_choices
                    
            except Exception as e:
                # En cas d'erreur, désactiver les champs avancés
                pass
    
    def clean(self):
        cleaned_data = super().clean()
        analysis_type = cleaned_data.get('analysis_type')
        
        # Validation selon le type d'analyse
        if analysis_type == 'regression':
            target = cleaned_data.get('target_column')
            features = cleaned_data.get('feature_columns')
            
            if not target:
                raise forms.ValidationError("Une colonne cible est requise pour la régression.")
            if not features:
                raise forms.ValidationError("Au moins une colonne explicative est requise pour la régression.")
            if target in features:
                raise forms.ValidationError("La colonne cible ne peut pas être dans les colonnes explicatives.")
        
        elif analysis_type == 'timeseries':
            date_col = cleaned_data.get('date_column')
            value_col = cleaned_data.get('value_column')
            
            if not date_col or not value_col:
                raise forms.ValidationError("Les colonnes de date et de valeur sont requises pour l'analyse temporelle.")
        
        return cleaned_data
    
    def get_parameters(self):
        """Construire les paramètres selon le type d'analyse"""
        if not self.is_valid():
            return {}
        
        analysis_type = self.cleaned_data['analysis_type']
        parameters = {}
        
        if analysis_type == 'regression':
            parameters = {
                'target_column': self.cleaned_data.get('target_column'),
                'feature_columns': list(self.cleaned_data.get('feature_columns', []))
            }
        elif analysis_type == 'clustering':
            parameters = {
                'n_clusters': self.cleaned_data.get('n_clusters', 3),
                'columns': []  # Utiliser toutes les colonnes numériques par défaut
            }
        elif analysis_type == 'timeseries':
            parameters = {
                'date_column': self.cleaned_data.get('date_column'),
                'value_column': self.cleaned_data.get('value_column')
            }
        
        return parameters