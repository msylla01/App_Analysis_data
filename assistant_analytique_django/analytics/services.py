import pandas as pd
import numpy as np
import json
from django.utils import timezone
from .models import Analysis, Dataset
import os
from .visualization_service import VisualizationService

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types NumPy"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

class DataAnalysisService:
    """Service pour effectuer les analyses de données"""
    
    def load_dataset(self, dataset):
        """Charger un dataset depuis un fichier"""
        file_path = dataset.file_path.path
        
        if dataset.dataset_type == 'csv':
            return pd.read_csv(file_path)
        elif dataset.dataset_type == 'excel':
            return pd.read_excel(file_path)
        elif dataset.dataset_type == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Type de fichier non supporté: {dataset.dataset_type}")
    
    def get_dataset_preview(self, dataset, rows=10):
        """Obtenir un aperçu du dataset"""
        try:
            df = self.load_dataset(dataset)
            
            preview = {
                'shape': [int(df.shape[0]), int(df.shape[1])],
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'head': self._convert_to_serializable(df.head(rows).to_dict('records')),
                'info': {
                    'memory_usage': int(df.memory_usage(deep=True).sum()),
                    'null_counts': {col: int(count) for col, count in df.isnull().sum().items()},
                }
            }
            
            return preview
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du dataset: {str(e)}")
    
    def validate_dataset_for_analysis(self, df, analysis_type):
        """Valider que le dataset est compatible avec le type d'analyse"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Informations de base sur le dataset
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        total_missing = df.isnull().sum().sum()
        
        # Validation générale
        if len(df) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Le dataset est vide")
            return validation_results
        
        if len(df.columns) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Le dataset n'a aucune colonne")
            return validation_results
        
        # Validations spécifiques par type d'analyse
        if analysis_type == 'correlation':
            if len(numeric_columns) < 2:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"L'analyse de corrélation nécessite au moins 2 colonnes numériques. "
                    f"Trouvées: {len(numeric_columns)} colonnes numériques"
                )
                if len(numeric_columns) > 0:
                    validation_results['recommendations'].append(
                        f"Colonnes numériques disponibles: {', '.join(numeric_columns)}"
                    )
                else:
                    validation_results['recommendations'].append(
                        "Aucune colonne numérique détectée. Vérifiez le format de vos données."
                    )
            else:
                # Vérifier s'il y a suffisamment de données non-nulles
                numeric_df = df[numeric_columns].dropna()
                if len(numeric_df) < 5:
                    validation_results['warnings'].append(
                        f"Peu de données valides pour la corrélation ({len(numeric_df)} lignes après suppression des valeurs manquantes)"
                    )
        
        elif analysis_type == 'descriptive':
            if len(numeric_columns) == 0 and len(categorical_columns) == 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Aucune colonne analysable trouvée")
            elif len(numeric_columns) == 0:
                validation_results['warnings'].append("Aucune colonne numérique pour les statistiques descriptives")
            elif len(categorical_columns) == 0:
                validation_results['warnings'].append("Aucune colonne catégorielle trouvée")
        
        # Avertissements généraux
        if total_missing > len(df) * len(df.columns) * 0.3:  # Plus de 30% de valeurs manquantes
            validation_results['warnings'].append(
                f"Beaucoup de valeurs manquantes ({total_missing}) - cela peut affecter les résultats"
            )
        
        return validation_results
    
    def _convert_to_serializable(self, obj):
        """Convertir les objets NumPy/Pandas en types sérialisables JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def run_analysis(self, analysis):
        """Exécuter une analyse selon son type"""
        try:
            analysis.status = 'running'
            analysis.save()
            
            df = self.load_dataset(analysis.dataset)
            
            # Valider le dataset avant l'analyse
            validation = self.validate_dataset_for_analysis(df, analysis.analysis_type)
            
            if not validation['is_valid']:
                raise ValueError(f"Validation échouée: {'; '.join(validation['errors'])}")
            
            # Exécuter l'analyse appropriée
            if analysis.analysis_type == 'descriptive':
                results = self._descriptive_analysis(df)
            elif analysis.analysis_type == 'correlation':
                results = self._correlation_analysis(df)
            elif analysis.analysis_type == 'regression':
                results = self._regression_analysis(df, analysis.parameters)
            elif analysis.analysis_type == 'clustering':
                results = self._clustering_analysis(df, analysis.parameters)
            elif analysis.analysis_type == 'timeseries':
                results = self._timeseries_analysis(df, analysis.parameters)
            else:
                raise ValueError(f"Type d'analyse non supporté: {analysis.analysis_type}")
            
            # Ajouter les informations de validation aux résultats
            results['validation'] = validation
            
            # Convertir tous les résultats en types sérialisables
            analysis.results = self._convert_to_serializable(results)
            analysis.status = 'completed'
            analysis.completed_at = timezone.now()
            analysis.save()
            
            # Créer automatiquement les visualisations
            try:
                viz_service = VisualizationService()
                visualizations = viz_service.create_visualizations_for_analysis(analysis)
                print(f"Créé {len(visualizations)} visualisations pour l'analyse {analysis.id}")
            except Exception as e:
                print(f"Erreur lors de la création des visualisations: {str(e)}")
            
        except Exception as e:
            analysis.status = 'failed'
            analysis.results = {'error': str(e)}
            analysis.save()
            raise
    
    def _descriptive_analysis(self, df):
        """Analyse descriptive des données"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        results = {
            'summary': {
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'numeric_columns': int(len(numeric_columns)),
                'categorical_columns': int(len(categorical_columns)),
                'missing_values': int(df.isnull().sum().sum()),
                'numeric_column_names': list(numeric_columns),
                'categorical_column_names': list(categorical_columns),
            },
            'numeric_stats': {},
            'categorical_stats': {},
        }
        
        # Statistiques numériques
        if len(numeric_columns) > 0:
            desc_stats = df[numeric_columns].describe()
            stats_dict = {}
            for col in desc_stats.columns:
                stats_dict[col] = {
                    'count': float(desc_stats.loc['count', col]),
                    'mean': float(desc_stats.loc['mean', col]),
                    'std': float(desc_stats.loc['std', col]),
                    'min': float(desc_stats.loc['min', col]),
                    'q25': float(desc_stats.loc['25%', col]),
                    'q50': float(desc_stats.loc['50%', col]),
                    'q75': float(desc_stats.loc['75%', col]),
                    'max': float(desc_stats.loc['max', col]),
                    'median': float(desc_stats.loc['50%', col]),
                }
            results['numeric_stats'] = stats_dict
        
        # Statistiques catégorielles
        for col in categorical_columns:
            try:
                value_counts = df[col].value_counts().head(10)
                results['categorical_stats'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_frequent': {str(k): int(v) for k, v in value_counts.items()},
                }
            except Exception as e:
                results['categorical_stats'][col] = {
                    'unique_values': 0,
                    'most_frequent': {},
                    'error': str(e)
                }
        
        return results
    
    def _correlation_analysis(self, df):
        """Analyse de corrélation"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Validation renforcée
        if len(numeric_df.columns) < 2:
            available_cols = list(df.columns)
            numeric_cols = list(numeric_df.columns)
            
            error_msg = (
                f"Au moins 2 colonnes numériques sont nécessaires pour l'analyse de corrélation. "
                f"Colonnes disponibles: {len(available_cols)} "
                f"({', '.join(available_cols[:5])}{'...' if len(available_cols) > 5 else ''}). "
                f"Colonnes numériques: {len(numeric_cols)} "
                f"({', '.join(numeric_cols) if numeric_cols else 'aucune'}). "
                f"Suggestion: Vérifiez que vos données numériques sont au bon format."
            )
            raise ValueError(error_msg)
        
        # Supprimer les colonnes avec toutes les valeurs identiques (variance = 0)
        numeric_df_clean = numeric_df.loc[:, numeric_df.var() != 0]
        
        if len(numeric_df_clean.columns) < 2:
            raise ValueError(
                f"Après nettoyage, pas assez de colonnes variables pour la corrélation. "
                f"Colonnes avec variance non-nulle: {list(numeric_df_clean.columns)}"
            )
        
        correlation_matrix = numeric_df_clean.corr()
        
        # Remplacer les NaN par 0 pour éviter les erreurs de sérialisation
        correlation_matrix = correlation_matrix.fillna(0)
        
        # Trouver les corrélations les plus fortes
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                # Vérifier que la valeur n'est pas NaN
                if not pd.isna(corr_value):
                    correlations.append({
                        'variable1': str(col1),
                        'variable2': str(col2),
                        'correlation': float(corr_value)
                    })
        
        # Trier par valeur absolue de corrélation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Convertir la matrice de corrélation en dictionnaire sérialisable
        correlation_dict = {}
        for col1 in correlation_matrix.columns:
            correlation_dict[str(col1)] = {}
            for col2 in correlation_matrix.columns:
                value = correlation_matrix.loc[col1, col2]
                correlation_dict[str(col1)][str(col2)] = float(value) if not pd.isna(value) else 0.0
        
        return {
            'correlation_matrix': correlation_dict,
            'strong_correlations': correlations[:10],
            'summary': {
                'variables_analyzed': int(len(numeric_df_clean.columns)),
                'original_numeric_columns': int(len(numeric_df.columns)),
                'removed_constant_columns': int(len(numeric_df.columns) - len(numeric_df_clean.columns)),
                'strongest_correlation': correlations[0] if correlations else None,
                'analyzed_columns': list(numeric_df_clean.columns),
            }
        }
    
    # ... (reste des méthodes inchangées)
    
    def _regression_analysis(self, df, parameters):
        """Analyse de régression simple"""
        target = parameters.get('target_column')
        features = parameters.get('feature_columns', [])
        
        if not target or not features:
            raise ValueError("Colonnes cible et caractéristiques requises pour la régression")
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Préparer les données
            X = df[features].select_dtypes(include=[np.number])
            y = df[target]
            
            # Supprimer les lignes avec des valeurs manquantes
            data = pd.concat([X, y], axis=1).dropna()
            X_clean = data[features]
            y_clean = data[target]
            
            if len(X_clean) < 5:
                raise ValueError("Pas assez de données pour effectuer une régression")
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            # Entraînement du modèle
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            return {
                'target': str(target),
                'features': [str(f) for f in features],
                'r2_score': float(r2_score(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'coefficients': {str(features[i]): float(coef) for i, coef in enumerate(model.coef_)},
                'intercept': float(model.intercept_),
                'n_samples': int(len(X_clean)),
            }
            
        except ImportError:
            return {
                'message': 'Scikit-learn non installé. Régression en cours de développement',
                'target': str(target),
                'features': [str(f) for f in features],
            }
    
    def _clustering_analysis(self, df, parameters):
        """Analyse de clustering"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            n_clusters = parameters.get('n_clusters', 3)
            columns = parameters.get('columns', [])
            
            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(columns) < 2:
                raise ValueError("Au moins 2 colonnes numériques sont nécessaires pour le clustering")
            
            data = df[columns].select_dtypes(include=[np.number]).dropna()
            
            if len(data) < n_clusters:
                raise ValueError(f"Pas assez de données pour créer {n_clusters} clusters")
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=42)
            clusters = kmeans.fit_predict(data_scaled)
            
            cluster_stats = {}
            for i in range(int(n_clusters)):
                cluster_data = data[clusters == i]
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(len(cluster_data)),
                    'percentage': float(len(cluster_data) / len(data) * 100),
                    'means': {col: float(cluster_data[col].mean()) for col in columns}
                }
            
            return {
                'n_clusters': int(n_clusters),
                'columns_used': [str(col) for col in columns],
                'total_samples': int(len(data)),
                'cluster_stats': cluster_stats,
                'inertia': float(kmeans.inertia_),
            }
            
        except ImportError:
            return {
                'message': 'Scikit-learn non installé. Clustering en cours de développement',
                'n_clusters': int(parameters.get('n_clusters', 3)),
            }
    
    def _timeseries_analysis(self, df, parameters):
        """Analyse de séries temporelles"""
        date_column = parameters.get('date_column')
        value_column = parameters.get('value_column')
        
        if not date_column or not value_column:
            return {
                'message': 'Analyse de séries temporelles en cours de développement',
                'date_column': str(date_column) if date_column else None,
                'value_column': str(value_column) if value_column else None,
            }
        
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            ts_data = df[[date_column, value_column]].dropna()
            ts_data = ts_data.sort_values(date_column)
            
            return {
                'date_column': str(date_column),
                'value_column': str(value_column),
                'date_range': {
                    'start': ts_data[date_column].min().isoformat(),
                    'end': ts_data[date_column].max().isoformat(),
                },
                'value_stats': {
                    'mean': float(ts_data[value_column].mean()),
                    'std': float(ts_data[value_column].std()),
                    'min': float(ts_data[value_column].min()),
                    'max': float(ts_data[value_column].max()),
                },
                'n_observations': int(len(ts_data)),
            }
            
        except Exception as e:
            return {
                'message': f'Erreur dans l\'analyse de séries temporelles: {str(e)}',
                'date_column': str(date_column),
                'value_column': str(value_column),
            }