from django import forms
from analytics.models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file_path', 'dataset_type', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Nom du dataset'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Description optionnelle du dataset'
            }),
            'file_path': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls,.json'
            }),
            'dataset_type': forms.Select(attrs={
                'class': 'form-select'
            }),
            'is_public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'].required = True
        self.fields['file_path'].required = True
        self.fields['dataset_type'].required = True