from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, UserProfile

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    company = forms.CharField(max_length=100, required=False)
    role = forms.CharField(max_length=50, required=False)
    
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'company', 'role', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            
        self.fields['username'].widget.attrs['placeholder'] = 'Nom d\'utilisateur'
        self.fields['email'].widget.attrs['placeholder'] = 'Email'
        self.fields['company'].widget.attrs['placeholder'] = 'Entreprise (optionnel)'
        self.fields['role'].widget.attrs['placeholder'] = 'Rôle (optionnel)'

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('bio', 'avatar')
        widgets = {
            'bio': forms.Textarea(attrs={
                'class': 'form-control', 
                'rows': 4,
                'placeholder': 'Parlez-nous un peu de vous...'
            }),
            'avatar': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
        }
    
    def clean_avatar(self):
        avatar = self.cleaned_data.get('avatar')
        
        if avatar:
            # Vérifier la taille du fichier (max 2MB)
            if avatar.size > 2 * 1024 * 1024:
                raise forms.ValidationError('La taille de l\'image ne doit pas dépasser 2 MB.')
            
            # Vérifier le type de fichier
            if not avatar.content_type.startswith('image/'):
                raise forms.ValidationError('Le fichier doit être une image.')
        
        return avatar