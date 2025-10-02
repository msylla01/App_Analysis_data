from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Récupère un élément d'un dictionnaire par sa clé"""
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None

@register.filter
def add_string(value, arg):
    """Concatène deux chaînes"""
    try:
        return str(value) + str(arg)
    except:
        return value

@register.filter
def multiply(value, arg):
    """Multiplie une valeur par un argument"""
    try:
        return float(value) * float(arg)
    except:
        return value

@register.filter
def percentage(part, total):
    """Calcule un pourcentage"""
    try:
        if total == 0:
            return 0
        return round((float(part) / float(total)) * 100, 2)
    except:
        return 0

@register.filter
def json_pretty(value):
    """Formate un JSON de manière lisible"""
    try:
        if isinstance(value, str):
            data = json.loads(value)
        else:
            data = value
        return mark_safe(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        return value