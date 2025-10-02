import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from .models import Analysis, Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service complet pour créer et gérer des visualisations"""
    
    # Palettes de couleurs
    COLOR_PALETTES = {
        'default': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384'],
        'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#F79256'],
        'nature': ['#2D5016', '#61A756', '#A2C523', '#D4E157', '#8BC34A', '#4CAF50'],
        'corporate': ['#1f4e79', '#2e86ab', '#a23b72', '#f18f01', '#c73e1d'],
        'pastel': ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E1BAFF'],
        'vibrant': ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'],
    }
    
    def __init__(self):
        self.default_options = {
            'responsive': True,
            'maintainAspectRatio': False,
            'interaction': {
                'intersect': False,
                'mode': 'index'
            },
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'backgroundColor': 'rgba(0,0,0,0.8)',
                    'titleColor': 'white',
                    'bodyColor': 'white',
                    'borderColor': 'rgba(255,255,255,0.1)',
                    'borderWidth': 1
                }
            },
            'animation': {
                'duration': 1000,
                'easing': 'easeInOutQuart'
            }
        }
    
    def create_visualizations_for_analysis(self, analysis):
        """Créer automatiquement des visualisations selon le type d'analyse"""
        visualizations = []
        
        try:
            logger.info(f"Création des visualisations pour l'analyse {analysis.id}")
            
            # Supprimer les anciennes visualisations
            old_count = Visualization.objects.filter(analysis=analysis).count()
            if old_count > 0:
                Visualization.objects.filter(analysis=analysis).delete()
                logger.info(f"Supprimé {old_count} anciennes visualisations")
            
            if not analysis.results:
                logger.warning(f"Aucun résultat trouvé pour l'analyse {analysis.id}")
                return visualizations
            
            if analysis.analysis_type == 'descriptive':
                visualizations.extend(self._create_descriptive_visualizations(analysis))
            elif analysis.analysis_type == 'correlation':
                visualizations.extend(self._create_correlation_visualizations(analysis))
            elif analysis.analysis_type == 'regression':
                visualizations.extend(self._create_regression_visualizations(analysis))
            elif analysis.analysis_type == 'clustering':
                visualizations.extend(self._create_clustering_visualizations(analysis))
            
            logger.info(f"Créé {len(visualizations)} visualisations avec succès")
            return visualizations
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des visualisations: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_descriptive_visualizations(self, analysis):
        """Créer des visualisations pour l'analyse descriptive"""
        visualizations = []
        results = analysis.results
        
        try:
            # 1. Graphique de résumé du dataset
            if results.get('summary'):
                viz = self._create_dataset_summary_chart(analysis, results['summary'])
                if viz:
                    visualizations.append(viz)
            
            # 2. Distribution des variables numériques
            if results.get('numeric_stats'):
                viz = self._create_numeric_distribution_chart(analysis, results['numeric_stats'])
                if viz:
                    visualizations.append(viz)
                
                # Box plot des variables numériques
                viz = self._create_numeric_boxplot(analysis, results['numeric_stats'])
                if viz:
                    visualizations.append(viz)
            
            # 3. Graphiques des variables catégorielles
            if results.get('categorical_stats'):
                vizs = self._create_categorical_charts(analysis, results['categorical_stats'])
                visualizations.extend(vizs)
            
            # 4. Graphique des valeurs manquantes
            if results.get('summary', {}).get('missing_values', 0) > 0:
                viz = self._create_missing_values_chart(analysis, results)
                if viz:
                    visualizations.append(viz)
                    
        except Exception as e:
            logger.error(f"Erreur dans _create_descriptive_visualizations: {str(e)}")
        
        return visualizations
    
    def _create_dataset_summary_chart(self, analysis, summary):
        """Créer un graphique de résumé du dataset"""
        try:
            data = {
                "labels": ["Colonnes Numériques", "Colonnes Catégorielles", "Valeurs Manquantes"],
                "datasets": [{
                    "label": "Nombre",
                    "data": [
                        int(summary.get('numeric_columns', 0)),
                        int(summary.get('categorical_columns', 0)),
                        int(summary.get('missing_values', 0))
                    ],
                    "backgroundColor": self.COLOR_PALETTES['professional'][:3],
                    "borderColor": self.COLOR_PALETTES['professional'][:3],
                    "borderWidth": 2,
                    "hoverBackgroundColor": [c + '80' for c in self.COLOR_PALETTES['professional'][:3]],
                }]
            }
            
            config = {
                "type": "doughnut",
                "options": {
                    **self.default_options,
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Composition du Dataset",
                            "font": {"size": 18, "weight": "bold"},
                            "padding": 20
                        },
                        "legend": {
                            "display": True,
                            "position": "bottom"
                        }
                    },
                    "cutout": "50%"
                }
            }
            
            viz = Visualization.objects.create(
                title="Composition du Dataset",
                description=f"Répartition des {summary.get('total_columns', 0)} colonnes du dataset ({summary.get('total_rows', 0)} lignes)",
                analysis=analysis,
                chart_type='doughnut',
                chart_data=data,
                chart_config=config,
                tags="résumé, composition, overview"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création graphique résumé: {str(e)}")
            return None
    
    def _create_numeric_distribution_chart(self, analysis, numeric_stats):
        """Créer un graphique de distribution des variables numériques"""
        try:
            if not numeric_stats:
                return None
            
            variables = list(numeric_stats.keys())[:6]  # Max 6 variables
            colors = self.COLOR_PALETTES['vibrant']
            
            datasets = []
            for i, var in enumerate(variables):
                stats = numeric_stats[var]
                datasets.append({
                    "label": var,
                    "data": [
                        stats.get('min', 0),
                        stats.get('q25', 0),
                        stats.get('median', 0),
                        stats.get('q75', 0),
                        stats.get('max', 0)
                    ],
                    "borderColor": colors[i % len(colors)],
                    "backgroundColor": colors[i % len(colors)] + '20',
                    "borderWidth": 3,
                    "fill": True,
                    "tension": 0.4
                })
            
            data = {
                "labels": ["Minimum", "Q1 (25%)", "Médiane", "Q3 (75%)", "Maximum"],
                "datasets": datasets
            }
            
            config = {
                "type": "radar",
                "options": {
                    **self.default_options,
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Distribution des Variables Numériques",
                            "font": {"size": 16, "weight": "bold"}
                        }
                    },
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "grid": {"color": "rgba(0,0,0,0.1)"},
                            "angleLines": {"color": "rgba(0,0,0,0.1)"},
                            "pointLabels": {"font": {"size": 12}}
                        }
                    }
                }
            }
            
            viz = Visualization.objects.create(
                title="Distribution des Variables Numériques",
                description=f"Radar chart des quartiles pour {len(variables)} variables numériques",
                analysis=analysis,
                chart_type='radar',
                chart_data=data,
                chart_config=config,
                tags="distribution, quartiles, numérique"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création distribution: {str(e)}")
            return None
    
    def _create_numeric_boxplot(self, analysis, numeric_stats):
        """Créer un box plot des variables numériques"""
        try:
            if not numeric_stats:
                return None
            
            variables = list(numeric_stats.keys())[:5]
            colors = self.COLOR_PALETTES['corporate']
            
            datasets = []
            for i, var in enumerate(variables):
                stats = numeric_stats[var]
                datasets.append({
                    "label": var,
                    "data": [{
                        "x": var,
                        "y": [
                            stats.get('min', 0),
                            stats.get('q25', 0),
                            stats.get('median', 0),
                            stats.get('q75', 0),
                            stats.get('max', 0)
                        ]
                    }],
                    "backgroundColor": colors[i % len(colors)] + '60',
                    "borderColor": colors[i % len(colors)],
                    "borderWidth": 2
                })
            
            # Simuler un box plot avec des barres
            box_data = {
                "labels": variables,
                "datasets": [{
                    "label": "Médiane",
                    "data": [numeric_stats[var].get('median', 0) for var in variables],
                    "backgroundColor": self.COLOR_PALETTES['professional'][0],
                    "borderColor": self.COLOR_PALETTES['professional'][0],
                    "borderWidth": 2
                }, {
                    "label": "Moyenne",
                    "data": [numeric_stats[var].get('mean', 0) for var in variables],
                    "backgroundColor": self.COLOR_PALETTES['professional'][1],
                    "borderColor": self.COLOR_PALETTES['professional'][1],
                    "borderWidth": 2
                }]
            }
            
            config = {
                "type": "bar",
                "options": {
                    **self.default_options,
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Comparaison Médiane vs Moyenne",
                            "font": {"size": 16, "weight": "bold"}
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "grid": {"color": "rgba(0,0,0,0.1)"},
                            "title": {"display": True, "text": "Valeurs"}
                        },
                        "x": {
                            "grid": {"display": False},
                            "title": {"display": True, "text": "Variables"}
                        }
                    }
                }
            }
            
            viz = Visualization.objects.create(
                title="Médiane vs Moyenne",
                description=f"Comparaison des tendances centrales pour {len(variables)} variables",
                analysis=analysis,
                chart_type='bar',
                chart_data=box_data,
                chart_config=config,
                tags="médiane, moyenne, comparaison"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création boxplot: {str(e)}")
            return None
    
    def _create_categorical_charts(self, analysis, categorical_stats):
        """Créer des graphiques pour les variables catégorielles"""
        visualizations = []
        variables = list(categorical_stats.keys())[:4]  # Max 4 variables
        
        for i, var in enumerate(variables):
            try:
                stats = categorical_stats[var]
                most_frequent = stats.get('most_frequent', {})
                
                if not most_frequent:
                    continue
                
                # Prendre les 8 valeurs les plus fréquentes
                items = list(most_frequent.items())[:8]
                other_count = sum([count for key, count in list(most_frequent.items())[8:]])
                
                if other_count > 0:
                    items.append(('Autres', other_count))
                
                if not items:
                    continue
                
                labels = [str(item[0]) for item in items]
                values = [int(item[1]) for item in items]
                
                # Utiliser différents types de graphiques
                chart_types = ['pie', 'doughnut', 'polarArea', 'bar']
                chart_type = chart_types[i % len(chart_types)]
                
                if chart_type in ['pie', 'doughnut', 'polarArea']:
                    data = {
                        "labels": labels,
                        "datasets": [{
                            "data": values,
                            "backgroundColor": self.COLOR_PALETTES['vibrant'][:len(values)],
                            "borderColor": '#ffffff',
                            "borderWidth": 2,
                            "hoverBorderWidth": 3
                        }]
                    }
                    
                    config_options = {
                        **self.default_options,
                        "plugins": {
                            **self.default_options['plugins'],
                            "title": {
                                "display": True,
                                "text": f"Distribution - {var}",
                                "font": {"size": 14, "weight": "bold"}
                            },
                            "legend": {
                                "position": "right",
                                "labels": {"usePointStyle": True, "padding": 15}
                            }
                        }
                    }
                    
                    if chart_type == 'doughnut':
                        config_options["cutout"] = "40%"
                    
                else:  # bar chart
                    data = {
                        "labels": labels,
                        "datasets": [{
                            "label": f"Fréquence - {var}",
                            "data": values,
                            "backgroundColor": self.COLOR_PALETTES['nature'][:len(values)],
                            "borderColor": self.COLOR_PALETTES['nature'][:len(values)],
                            "borderWidth": 1
                        }]
                    }
                    
                    config_options = {
                        **self.default_options,
                        "plugins": {
                            **self.default_options['plugins'],
                            "title": {
                                "display": True,
                                "text": f"Fréquences - {var}",
                                "font": {"size": 14, "weight": "bold"}
                            },
                            "legend": {"display": False}
                        },
                        "scales": {
                            "y": {
                                "beginAtZero": True,
                                "title": {"display": True, "text": "Fréquence"}
                            },
                            "x": {
                                "title": {"display": True, "text": "Valeurs"}
                            }
                        }
                    }
                
                config = {
                    "type": chart_type,
                    "options": config_options
                }
                
                viz = Visualization.objects.create(
                    title=f"Distribution - {var}",
                    description=f"Répartition des valeurs pour {var} ({stats.get('unique_values', 0)} valeurs uniques)",
                    analysis=analysis,
                    chart_type=chart_type,
                    chart_data=data,
                    chart_config=config,
                    tags=f"catégoriel, distribution, {var.lower()}"
                )
                
                visualizations.append(viz)
                
            except Exception as e:
                logger.error(f"Erreur création graphique catégoriel {var}: {str(e)}")
                continue
        
        return visualizations
    
    def _create_missing_values_chart(self, analysis, results):
        """Créer un graphique des valeurs manquantes"""
        try:
            summary = results.get('summary', {})
            
            # Données simulées pour les valeurs manquantes par colonne
            missing_data = {
                "labels": ["Complètes", "Avec valeurs manquantes"],
                "datasets": [{
                    "data": [
                        summary.get('total_rows', 0) - summary.get('missing_values', 0),
                        summary.get('missing_values', 0)
                    ],
                    "backgroundColor": ['#2ECC71', '#E74C3C'],
                    "borderColor": ['#27AE60', '#C0392B'],
                    "borderWidth": 2
                }]
            }
            
            config = {
                "type": "pie",
                "options": {
                    **self.default_options,
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Qualité des Données",
                            "font": {"size": 16, "weight": "bold"}
                        },
                        "legend": {
                            "position": "bottom"
                        }
                    }
                }
            }
            
            viz = Visualization.objects.create(
                title="Qualité des Données",
                description=f"Répartition des données complètes vs manquantes ({summary.get('missing_values', 0)} valeurs manquantes)",
                analysis=analysis,
                chart_type='pie',
                chart_data=missing_data,
                chart_config=config,
                tags="qualité, valeurs manquantes, nettoyage"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création graphique valeurs manquantes: {str(e)}")
            return None
    
    def _create_correlation_visualizations(self, analysis):
        """Créer des visualisations pour l'analyse de corrélation"""
        visualizations = []
        results = analysis.results
        
        try:
            # 1. Graphique en barres des corrélations fortes
            if results.get('strong_correlations'):
                viz = self._create_correlation_bar_chart(analysis, results['strong_correlations'])
                if viz:
                    visualizations.append(viz)
            
            # 2. Heatmap de corrélation (version simplifiée)
            if results.get('correlation_matrix'):
                viz = self._create_correlation_heatmap_simple(analysis, results['correlation_matrix'])
                if viz:
                    visualizations.append(viz)
                    
        except Exception as e:
            logger.error(f"Erreur dans _create_correlation_visualizations: {str(e)}")
        
        return visualizations
    
    def _create_correlation_bar_chart(self, analysis, strong_correlations):
        """Créer un graphique en barres des corrélations"""
        try:
            top_correlations = strong_correlations[:10]
            
            if not top_correlations:
                return None
            
            labels = [f"{corr['variable1']} ↔ {corr['variable2']}" for corr in top_correlations]
            values = [abs(float(corr['correlation'])) for corr in top_correlations]
            colors = ['#2ECC71' if float(corr['correlation']) > 0 else '#E74C3C' for corr in top_correlations]
            
            data = {
                "labels": labels,
                "datasets": [{
                    "label": "Force de corrélation",
                    "data": values,
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1,
                    "barThickness": 30
                }]
            }
            
            config = {
                "type": "bar",
                "options": {
                    **self.default_options,
                    "indexAxis": "y",
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Corrélations les Plus Fortes",
                            "font": {"size": 16, "weight": "bold"}
                        },
                        "legend": {"display": False},
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return 'Corrélation: ' + context.parsed.x.toFixed(3); }"
                            }
                        }
                    },
                    "scales": {
                        "x": {
                            "beginAtZero": True,
                            "max": 1,
                            "title": {"display": True, "text": "Force de corrélation (valeur absolue)"},
                            "grid": {"color": "rgba(0,0,0,0.1)"}
                        },
                        "y": {
                            "title": {"display": True, "text": "Paires de variables"},
                            "ticks": {"font": {"size": 10}}
                        }
                    }
                }
            }
            
            viz = Visualization.objects.create(
                title="Top Corrélations",
                description=f"Les {len(top_correlations)} corrélations les plus fortes entre variables",
                analysis=analysis,
                chart_type='horizontalBar',
                chart_data=data,
                chart_config=config,
                tags="corrélation, relations, variables"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création graphique corrélation: {str(e)}")
            return None
    
    def _create_correlation_heatmap_simple(self, analysis, correlation_matrix):
        """Créer une heatmap simplifiée des corrélations"""
        try:
            # Prendre un échantillon de la matrice pour la lisibilité
            variables = list(correlation_matrix.keys())[:8]  # Max 8 variables
            
            if len(variables) < 2:
                return None
            
            # Créer des données pour un graphique radar des corrélations moyennes
            avg_correlations = []
            for var in variables:
                correlations_for_var = [
                    abs(correlation_matrix[var][other_var]) 
                    for other_var in variables 
                    if other_var != var
                ]
                avg_corr = sum(correlations_for_var) / len(correlations_for_var) if correlations_for_var else 0
                avg_correlations.append(avg_corr)
            
            data = {
                "labels": variables,
                "datasets": [{
                    "label": "Corrélation moyenne",
                    "data": avg_correlations,
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                    "fill": True
                }]
            }
            
            config = {
                "type": "radar",
                "options": {
                    **self.default_options,
                    "plugins": {
                        **self.default_options['plugins'],
                        "title": {
                            "display": True,
                            "text": "Profil de Corrélation des Variables",
                            "font": {"size": 16, "weight": "bold"}
                        }
                    },
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 1,
                            "ticks": {"stepSize": 0.2}
                        }
                    }
                }
            }
            
            viz = Visualization.objects.create(
                title="Profil de Corrélation",
                description=f"Corrélation moyenne de chaque variable avec les autres ({len(variables)} variables)",
                analysis=analysis,
                chart_type='radar',
                chart_data=data,
                chart_config=config,
                tags="corrélation, profil, heatmap"
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création heatmap: {str(e)}")
            return None
    
    def _create_regression_visualizations(self, analysis):
        """Créer des visualisations pour l'analyse de régression"""
        visualizations = []
        # À implémenter selon les besoins spécifiques de régression
        return visualizations
    
    def _create_clustering_visualizations(self, analysis):
        """Créer des visualisations pour l'analyse de clustering"""
        visualizations = []
        # À implémenter selon les besoins spécifiques de clustering
        return visualizations
    
    def create_custom_visualization(self, analysis, title, description, chart_type, data, config, tags=""):
        """Créer une visualisation personnalisée"""
        try:
            # Validation des données
            if not self._validate_chart_data(data, chart_type):
                raise ValueError("Données invalides pour le type de graphique")
            
            # Fusionner avec les options par défaut
            final_config = {
                "type": chart_type,
                "options": {**self.default_options, **(config.get('options', {}))}
            }
            
            viz = Visualization.objects.create(
                title=title,
                description=description,
                analysis=analysis,
                chart_type=chart_type,
                chart_data=data,
                chart_config=final_config,
                tags=tags
            )
            
            return viz
            
        except Exception as e:
            logger.error(f"Erreur création visualisation personnalisée: {str(e)}")
            return None
    
    def _validate_chart_data(self, data, chart_type):
        """Valider que les données correspondent au type de graphique"""
        try:
            if not isinstance(data, dict):
                return False
            
            if 'labels' not in data or 'datasets' not in data:
                return False
            
            if not isinstance(data['labels'], list) or not isinstance(data['datasets'], list):
                return False
            
            if len(data['datasets']) == 0:
                return False
            
            # Validations spécifiques par type
            for dataset in data['datasets']:
                if 'data' not in dataset or not isinstance(dataset['data'], list):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def update_visualization_style(self, visualization, style_config):
        """Mettre à jour le style d'une visualisation"""
        try:
            current_config = visualization.chart_config
            
            # Fusionner le nouveau style
            if 'options' not in current_config:
                current_config['options'] = {}
            
            current_config['options'].update(style_config)
            
            visualization.chart_config = current_config
            visualization.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur mise à jour style: {str(e)}")
            return False