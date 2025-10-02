# App_Analysis_data
# Assistant Analytique Génie

Un assistant d'analyse de données moderne et interactif développé avec Django, permettant l'upload, l'analyse et la visualisation de données avec des graphiques interactifs.

## 🚀 Fonctionnalités

- **📊 Analyse de Données Avancée**
  - Analyse descriptive (statistiques, distributions)
  - Analyse de corrélation avec matrices interactives
  - Analyse de régression (en développement)
  - Clustering et analyse temporelle

- **📈 Visualisations Interactives**
  - Graphiques en barres, lignes, secteurs
  - Graphiques radar et en anneau
  - Heatmaps de corrélation
  - Mode plein écran et zoom
  - Export PNG/SVG/JSON

- **🔐 Gestion des Utilisateurs**
  - Authentification sécurisée
  - Profils utilisateur personnalisables
  - Gestion des datasets privés/publics

- **📁 Support Multi-Format**
  - CSV, Excel (.xlsx, .xls)
  - JSON
  - Upload par glisser-déposer

## 🛠️ Technologies Utilisées

- **Backend**: Django 4.2.7, Python 3.9+
- **Frontend**: Bootstrap 5, Chart.js, Plotly.js
- **Base de données**: PostgreSQL
- **Analyse**: Pandas, NumPy, Scikit-learn
- **API**: Django REST Framework

## 📋 Prérequis

- Python 3.9 ou supérieur
- PostgreSQL 12+
- Git
- pip (gestionnaire de packages Python)

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/msylla01/App_Analysis_data.git
cd App_Analysis_data


### 2. Créer un environnement virtuel
```bash
# Avec venv
python -m venv venv_django

# Activer l'environnement virtuel
# Sur Linux/macOS:
source venv_django/bin/activate
# Sur Windows:
# venv_django\Scripts\activate


3. Installer les dépendances
pip install -r requirements.txt

4. Configuration de la base de données
Créer la base de données PostgreSQL


-- Se connecter à PostgreSQL en tant que superutilisateur
sudo -u postgres psql

-- Créer la base de données et l'utilisateur
CREATE DATABASE assistant_analytique;
CREATE USER assistant_user WITH PASSWORD 'votre_mot_de_passe';
GRANT ALL PRIVILEGES ON DATABASE assistant_analytique TO assistant_user;
ALTER USER assistant_user CREATEDB;
\q


Configuration des variables d'environnement

Créez un fichier .env à la racine du projet :

# .env
DEBUG=True
SECRET_KEY=votre-clé-secrète-très-longue-et-complexe
DATABASE_URL=postgresql://assistant_user:votre_mot_de_passe@localhost:5432/assistant_analytique

# Configuration email (optionnel)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=votre-email@gmail.com
EMAIL_HOST_PASSWORD=votre-mot-de-passe-app

# Configuration Redis (pour Celery - optionnel)
REDIS_URL=redis://localhost:6379/0


5. Appliquer les migrations

# Créer et appliquer les migrations
python manage.py makemigrations
python manage.py migrate

# Créer les tables de visualisation si nécessaire
python manage.py makemigrations analytics
python manage.py migrate analytics

6. Créer un superutilisateur
python manage.py createsuperuser

7. Collecter les fichiers statiques
python manage.py collectstatic --noinput


🏃‍♂️ Démarrage
Démarrer le serveur de développement
python manage.py runserver

L'application sera accessible à l'adresse : http://127.0.0.1:8000/
