import openai
import pandas as pd
import numpy as np
import json
import re
import io
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
from decouple import config
from django.conf import settings
from .models import ChatSession, ChatMessage, DataAnalysisQuery, Dataset
import logging
import traceback
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataAnalysisAI:
    """Service d'IA pour l'analyse de données en langage naturel"""
    
    def __init__(self):
        try:
            # Nettoyer les variables d'environnement proxy qui causent des conflits
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            for var in proxy_vars:
                if var in os.environ:
                    del os.environ[var]
            
            # Configuration OpenAI sans proxy
            api_key = config('OPENAI_API_KEY', default='')
            if not api_key:
                raise Exception("OPENAI_API_KEY non configurée dans le fichier .env")
            
            # Essayer d'abord la nouvelle API OpenAI
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0,  # Timeout explicite
            )
            self.use_new_api = True
            logger.info("✅ OpenAI API v1.x initialisée avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion OpenAI: {str(e)}")
            try:
                # Fallback vers l'ancienne API
                openai.api_key = config('OPENAI_API_KEY')
                self.client = None
                self.use_new_api = False
                logger.info("⚠️ Utilisation de l'API OpenAI legacy (v0.x)")
            except Exception as e2:
                logger.error(f"❌ Toutes les APIs OpenAI ont échoué: {e2}")
                # Mode dégradé : on peut continuer sans OpenAI
                self.client = None
                self.use_new_api = False
                self.openai_available = False
                logger.warning("🔄 Mode dégradé activé : OpenAI non disponible")
                return
        
        self.openai_available = True
        self.model = config('OPENAI_MODEL', default='gpt-3.5-turbo')
        self.max_tokens = int(config('OPENAI_MAX_TOKENS', default=1500))
        
        # Prompts système optimisés
        self.system_prompts = {
            'analyst': """Vous êtes un expert en analyse de données avec Python, pandas, numpy, matplotlib et seaborn.

VOTRE RÔLE:
- Analyser les demandes en langage naturel
- Expliquer les analyses de manière claire
- Suggérer des analyses complémentaires

RÈGLES IMPORTANTES:
- Réponses en français
- Langage accessible mais technique
- Soyez précis et informatif
- Proposez des actions concrètes""",

            'code_generator': """Analysez la demande et classifiez-la.

Retournez un JSON valide avec:
{
    "type": "data_exploration|data_analysis|visualization|general_question",
    "intent": "description courte",
    "suggested_approach": "approche suggérée",
    "complexity": "low|medium|high"
}""",

            'interpreter': """Interprétez les résultats d'analyses de données.

VOTRE MISSION:
- Expliquer les résultats en termes métier
- Identifier les insights clés
- Suggérer des actions ou analyses complémentaires
- Utiliser un langage accessible
- Être objectif et factuel"""
        }
    
    async def process_user_message(self, session, user_message):
        """Traiter un message utilisateur et générer une réponse IA"""
        try:
            logger.info(f"📝 Traitement du message: {user_message[:100]}...")
            
            # Créer le message utilisateur
            user_msg = ChatMessage.objects.create(
                session=session,
                role='user',
                content=user_message,
                message_type='text'
            )
            
            # Si OpenAI n'est pas disponible, utiliser le mode dégradé
            if not hasattr(self, 'openai_available') or not self.openai_available:
                return await self._handle_fallback_mode(session, user_msg, user_message)
            
            # Analyser le type de requête
            query_analysis = await self._analyze_query(user_message, session.dataset)
            
            # Générer la réponse selon le type
            if query_analysis['type'] == 'data_analysis':
                response = await self._handle_data_analysis(session, user_msg, query_analysis)
            elif query_analysis['type'] == 'data_exploration':
                response = await self._handle_data_exploration(session, user_msg, query_analysis)
            elif query_analysis['type'] == 'visualization':
                response = await self._handle_visualization_request(session, user_msg, query_analysis)
            elif query_analysis['type'] == 'general_question':
                response = await self._handle_general_question(session, user_msg, query_analysis)
            else:
                response = await self._handle_fallback(session, user_msg)
            
            # Mettre à jour les compteurs
            session.message_count += 1
            if query_analysis['type'] in ['data_analysis', 'data_exploration']:
                session.analysis_count += 1
            session.save()
            
            return response
            
        except Exception as e:
            logger.error(f"🚫 Erreur lors du traitement du message: {str(e)}")
            traceback.print_exc()
            return await self._create_error_response(session, str(e))
    
    async def _handle_fallback_mode(self, session, user_msg, user_message):
        """Mode dégradé quand OpenAI n'est pas disponible"""
        logger.info("🔄 Mode dégradé activé - analyse heuristique")
        
        # Analyse heuristique basique
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['résumé', 'aperçu', 'explore', 'montre', 'structure']):
            return await self._handle_data_exploration(session, user_msg, {
                'type': 'data_exploration',
                'intent': 'Exploration des données',
                'suggested_approach': 'Analyse descriptive',
                'complexity': 'low'
            })
        elif any(word in message_lower for word in ['graphique', 'plot', 'visualise', 'chart']):
            return await self._handle_visualization_request(session, user_msg, {
                'type': 'visualization',
                'intent': 'Création de visualisation',
                'suggested_approach': 'Graphique',
                'complexity': 'medium'
            })
        else:
            return ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=f"""🤖 **Assistant en mode simplifié**

J'ai reçu votre message : "{user_message[:100]}..."

⚠️ **Note :** Le service IA avancé n'est temporairement pas disponible, mais je peux toujours vous aider !

**Ce que je peux faire :**
• 📊 Explorer la structure de vos données
• 📈 Calculer des statistiques descriptives
• 🔍 Analyser les colonnes disponibles
• 💡 Vous donner des conseils d'analyse

**Essayez ces commandes :**
• "Montre-moi mes données"
• "Résume mon dataset"
• "Quelles colonnes ai-je ?"

Comment puis-je vous aider avec vos données ?""",
                message_type='text'
            )
    
    async def _analyze_query(self, message, dataset):
        """Analyser le message pour déterminer le type de requête"""
        try:
            # Si OpenAI n'est pas disponible, analyse heuristique
            if not hasattr(self, 'openai_available') or not self.openai_available:
                return self._heuristic_analysis(message)
            
            # Obtenir des informations sur le dataset
            dataset_info = ""
            if dataset:
                try:
                    from .services import DataAnalysisService
                    service = DataAnalysisService()
                    df = service.load_dataset(dataset)
                    
                    dataset_info = f"""
Dataset: {dataset.name}
- Lignes: {len(df)}
- Colonnes: {len(df.columns)}
- Colonnes principales: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}
"""
                except Exception as e:
                    dataset_info = f"Dataset: {dataset.name} (erreur: {str(e)})"
            
            prompt = f"""Analysez cette demande d'analyse de données:

Message: "{message}"
{dataset_info}

Classifiez dans: data_analysis, data_exploration, visualization, general_question

{{
    "type": "category",
    "intent": "description courte",
    "suggested_approach": "approche",
    "complexity": "low|medium|high"
}}"""

            response = await self._call_openai(prompt, "code_generator")
            
            try:
                analysis = json.loads(response)
            except:
                # Fallback heuristique
                analysis = self._heuristic_analysis(message)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse requête: {str(e)}")
            return self._heuristic_analysis(message)
    
    def _heuristic_analysis(self, message):
        """Analyse heuristique sans IA"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['graphique', 'plot', 'visualise', 'chart']):
            return {"type": "visualization", "intent": "Création de visualisation", "suggested_approach": "Graphique", "complexity": "medium"}
        elif any(word in message_lower for word in ['résumé', 'aperçu', 'explore', 'montre', 'structure']):
            return {"type": "data_exploration", "intent": "Exploration des données", "suggested_approach": "Statistiques descriptives", "complexity": "low"}
        elif any(word in message_lower for word in ['corrél', 'analyse', 'test', 'régression']):
            return {"type": "data_analysis", "intent": "Analyse statistique", "suggested_approach": "Analyse avancée", "complexity": "high"}
        else:
            return {"type": "general_question", "intent": "Question générale", "suggested_approach": "Réponse informative", "complexity": "low"}
    
    async def _handle_data_exploration(self, session, user_msg, query_analysis):
        """Gérer une demande d'exploration de données"""
        try:
            if not session.dataset:
                return ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content="""📊 **Exploration de données**

Pour explorer vos données, je dois d'abord connaître le dataset ! 

**Étapes à suivre :**
1. 📁 Sélectionnez un dataset dans le panneau de droite
2. 📤 Ou uploadez un nouveau fichier
3. 🔍 Puis demandez-moi de l'analyser

**Ce que je peux faire :**
• 📈 Analyser la structure des données
• 🔢 Calculer des statistiques descriptives  
• ⚠️ Détecter les valeurs manquantes
• 🎯 Suggérer des analyses pertinentes

Sélectionnez d'abord un dataset !""",
                    message_type='text'
                )
            
            # Analyser le dataset
            from .services import DataAnalysisService
            service = DataAnalysisService()
            df = service.load_dataset(session.dataset)
            
            # Statistiques détaillées
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Calculer des métriques de qualité
            missing_values = df.isnull().sum().sum()
            missing_percent = (missing_values / (df.shape[0] * df.shape[1])) * 100
            
            summary = f"""📊 **Exploration complète de {session.dataset.name}**

**📐 Structure générale :**
• Lignes : {df.shape[0]:,}
• Colonnes : {df.shape[1]:,}
• Taille mémoire : ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

**📊 Types de données :**
• Numériques : {len(numeric_cols)} colonnes
• Catégorielles : {len(categorical_cols)} colonnes
• Booléennes : {len(df.select_dtypes(include=['bool']).columns)} colonnes

**⚠️ Qualité des données :**
• Valeurs manquantes : {missing_values:,} ({missing_percent:.1f}%)
• Lignes complètes : {df.dropna().shape[0]:,} ({(df.dropna().shape[0]/df.shape[0]*100):.1f}%)

**🔍 Colonnes numériques :**
{', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}

**📝 Colonnes catégorielles :**
{', '.join(categorical_cols[:8])}{'...' if len(categorical_cols) > 8 else ''}

**💡 Suggestions d'analyse :**
• Analysez les corrélations entre variables numériques
• Explorez la distribution des variables importantes
• Détectez les valeurs aberrantes
• Créez des visualisations descriptives

Que souhaitez-vous analyser en particulier ?"""
            
            return ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=summary,
                message_type='data_summary',
                structured_data={
                    'dataset_id': session.dataset.id,
                    'shape': df.shape,
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': missing_values
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur exploration: {str(e)}")
            return await self._create_error_response(session, str(e))
    
    async def _handle_general_question(self, session, user_msg, query_analysis):
        """Gérer une question générale"""
        try:
            # Si OpenAI n'est pas disponible, réponse générique
            if not hasattr(self, 'openai_available') or not self.openai_available:
                return ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=f"""🤖 **Réponse à votre question**

Vous avez demandé : "{user_msg.content}"

**En mode simplifié, je peux vous aider avec :**
• 📊 L'exploration de vos datasets
• 📈 L'explication des concepts d'analyse de données
• 🔍 L'identification des bonnes pratiques
• 💡 Des suggestions d'analyses adaptées à vos données

**Questions fréquentes :**
• "Comment analyser des corrélations ?"
• "Quels graphiques pour mes données ?"
• "Comment détecter des anomalies ?"
• "Comment interpréter mes résultats ?"

Pouvez-vous être plus spécifique sur ce que vous cherchez à analyser ?""",
                    message_type='text'
                )
            
            # Avec OpenAI
            context = ""
            if session.dataset:
                context = f"L'utilisateur travaille avec le dataset '{session.dataset.name}'."
            
            prompt = f"""Répondez à cette question sur l'analyse de données:

Question: "{user_msg.content}"
Contexte: {context}

Réponse en français, informative et pédagogique."""

            response_content = await self._call_openai(prompt, "analyst")
            
            return ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=response_content,
                message_type='text'
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur question générale: {str(e)}")
            return await self._create_error_response(session, str(e))
    
    async def _handle_data_analysis(self, session, user_msg, query_analysis):
        """Gérer une demande d'analyse de données"""
        try:
            if not session.dataset:
                return ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content="""🔬 **Analyse de données avancée**

Pour effectuer une analyse, j'ai besoin d'un dataset !

Sélectionnez d'abord un dataset, puis je pourrai vous aider avec :
• 📊 Analyses statistiques descriptives
• 🔗 Corrélations entre variables
• 📈 Analyses de tendances
• 🎯 Détection d'anomalies
• 📉 Tests statistiques""",
                    message_type='text'
                )
            
            response_content = f"""🔬 **Analyse demandée : {query_analysis.get('intent', 'Analyse générale')}**

**Votre demande :** "{user_msg.content}"

**Type d'analyse détecté :** {query_analysis.get('suggested_approach', 'Analyse standard')}
**Complexité :** {query_analysis.get('complexity', 'medium')} ⭐

**Dataset :** {session.dataset.name}

**🔄 Statut :** Le système d'analyse automatique est en cours de développement.

**💡 En attendant, je peux vous aider avec :**
• Exploration détaillée de vos données
• Suggestions d'analyses appropriées
• Conseils sur les méthodes statistiques
• Interprétation de résultats

**Suggestions spécifiques pour votre demande :**
• Commencez par explorer la structure de vos données
• Vérifiez la qualité et les valeurs manquantes
• Identifiez les variables clés pour votre analyse

Voulez-vous que je commence par explorer votre dataset ?"""
            
            return ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=response_content,
                message_type='analysis',
                structured_data={
                    'analysis_type': query_analysis.get('intent'),
                    'complexity': query_analysis.get('complexity'),
                    'dataset_id': session.dataset.id if session.dataset else None
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse: {str(e)}")
            return await self._create_error_response(session, str(e))
    
    async def _handle_visualization_request(self, session, user_msg, query_analysis):
        """Gérer une demande de visualisation"""
        try:
            if not session.dataset:
                return ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content="""📊 **Création de visualisations**

Pour créer des graphiques, j'ai besoin d'un dataset !

**Types de visualisations disponibles :**
• 📈 Graphiques en ligne (tendances)
• 📊 Histogrammes (distributions)
• 🔄 Graphiques circulaires (proportions)
• 📉 Nuages de points (corrélations)
• 📋 Matrices de corrélation

Sélectionnez d'abord un dataset !""",
                    message_type='text'
                )
            
            # Analyser le dataset pour suggérer des visualisations
            from .services import DataAnalysisService
            service = DataAnalysisService()
            df = service.load_dataset(session.dataset)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            suggestions = []
            if len(numeric_cols) >= 2:
                suggestions.append(f"• Nuage de points : {numeric_cols[0]} vs {numeric_cols[1]}")
                suggestions.append(f"• Matrice de corrélation entre variables numériques")
            
            if len(numeric_cols) >= 1:
                suggestions.append(f"• Histogramme de {numeric_cols[0]}")
                suggestions.append(f"• Boîte à moustaches pour détecter les outliers")
            
            if len(categorical_cols) >= 1:
                suggestions.append(f"• Graphique en barres de {categorical_cols[0]}")
            
            response_content = f"""📊 **Demande de visualisation**

**Votre demande :** "{user_msg.content}"
**Dataset :** {session.dataset.name} ({df.shape[0]} lignes × {df.shape[1]} colonnes)

**📈 Variables disponibles :**
• Numériques : {len(numeric_cols)} ({', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''})
• Catégorielles : {len(categorical_cols)} ({', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''})

**💡 Suggestions de visualisations :**
{chr(10).join(suggestions[:5])}

**🔄 Note :** La génération automatique de graphiques est en développement.

**En attendant :**
• Utilisez l'interface de visualisation classique
• Je peux vous conseiller sur le type de graphique approprié
• Demandez-moi des conseils sur la visualisation de données

Quel type de graphique vous intéresse le plus ?"""
            
            return ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=response_content,
                message_type='visualization',
                structured_data={
                    'dataset_id': session.dataset.id,
                    'numeric_columns': list(numeric_cols),
                    'categorical_columns': list(categorical_cols),
                    'suggestions': suggestions
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur visualisation: {str(e)}")
            return await self._create_error_response(session, str(e))
    
    async def _handle_fallback(self, session, user_msg):
        """Réponse de fallback"""
        return ChatMessage.objects.create(
            session=session,
            role='assistant',
            content="""🤔 **Je n'ai pas bien compris votre demande**

**Voici ce que je peux faire :**
• 📊 **Explorer vos données :** "Montre-moi la structure de mes données"
• 🔍 **Analyser les relations :** "Quelles sont les corrélations ?"
• 📈 **Conseiller sur les graphiques :** "Quel graphique pour mes données ?"
• 💡 **Répondre aux questions :** "Comment interpréter une corrélation ?"

**Exemples de questions :**
• "Résume mon dataset"
• "Combien ai-je de lignes et colonnes ?"
• "Quels types de données ai-je ?"
• "Comment détecter des anomalies ?"

Pouvez-vous reformuler votre question ou être plus spécifique ? 😊""",
            message_type='text'
        )
    
    async def _create_error_response(self, session, error_message):
        """Créer une réponse d'erreur"""
        return ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=f"""❌ **Une erreur s'est produite**

**Détails :** {error_message}

**Que faire :**
• Réessayez votre demande
• Vérifiez que votre dataset est accessible
• Reformulez votre question plus simplement
• Contactez le support si le problème persiste

Je reste disponible pour vous aider ! 😊""",
            message_type='error',
            structured_data={'error': error_message}
        )
    
    async def _call_openai(self, prompt, prompt_type="analyst"):
        """Appeler l'API OpenAI avec gestion d'erreurs robuste"""
        try:
            if not hasattr(self, 'openai_available') or not self.openai_available:
                raise Exception("OpenAI non disponible")
            
            system_prompt = self.system_prompts.get(prompt_type, self.system_prompts['analyst'])
            
            if self.use_new_api:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    timeout=30
                )
                return response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    request_timeout=30
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Erreur API OpenAI: {str(e)}")
            # Marquer OpenAI comme non disponible pour cette session
            self.openai_available = False
            raise e
    
    def get_analysis_suggestions(self, dataset):
        """Obtenir des suggestions d'analyse pour un dataset"""
        try:
            if not dataset:
                return [
                    "Sélectionnez d'abord un dataset",
                    "Uploadez un nouveau fichier",
                    "Explorez les datasets existants",
                    "Demandez de l'aide sur l'analyse"
                ]
            
            from .services import DataAnalysisService
            service = DataAnalysisService()
            df = service.load_dataset(dataset)
            
            suggestions = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Suggestions intelligentes basées sur les données
            suggestions.append("Montre-moi la structure de mes données")
            
            if len(numeric_cols) >= 2:
                suggestions.extend([
                    "Analyse les corrélations entre variables",
                    "Détecte les valeurs aberrantes"
                ])
            
            if len(numeric_cols) >= 1:
                suggestions.append("Calcule les statistiques descriptives")
            
            if len(categorical_cols) >= 1:
                suggestions.append("Analyse la distribution des catégories")
            
            suggestions.extend([
                "Vérifie la qualité des données",
                "Suggère des visualisations appropriées"
            ])
            
            return suggestions[:6]
            
        except Exception as e:
            logger.error(f"❌ Erreur suggestions: {str(e)}")
            return [
                "Explorez votre dataset",
                "Calculez des statistiques",
                "Analysez les corrélations",
                "Créez des visualisations"
            ]