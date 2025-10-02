from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserProfile

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['email', 'username', 'company', 'role', 'is_staff', 'created_at']
    list_filter = ['is_staff', 'is_active', 'created_at', 'role']
    search_fields = ['email', 'username', 'company']
    ordering = ['-created_at']
    
    fieldsets = UserAdmin.fieldsets + (
        ('Informations supplémentaires', {'fields': ('company', 'role')}),
    )

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UserProfile)