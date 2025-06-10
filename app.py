import streamlit as st
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re
import json
from typing import List, Tuple
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import html
import time

# Charger les variables d'environnement
load_dotenv()

# Configuration OpenAI - Version mise à jour
from openai import OpenAI

# Configuration de la clé API
def get_openai_client():
    """Initialise le client OpenAI de manière sécurisée"""
    try:
        # Essayer d'abord avec st.secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            # Puis avec les variables d'environnement
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            return OpenAI(api_key=api_key), api_key
        else:
            return None, None
    except Exception as e:
        st.error(f"Erreur lors de la récupération de la clé API : {e}")
        return None, None

class TextAnalyzer:
    def __init__(self, openai_api_key: str):
        # Configuration OpenAI - Focus sur GPT-4.1 avec fallback
        self.client = OpenAI(api_key=openai_api_key)
        
        # Essayer d'abord GPT-4.1, puis se rabattre sur GPT-4o-mini si non disponible
        self.model_priority = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o-mini']
        self.model = self._get_available_model()
        
        # Définir l'encoding approprié
        self.encoding_name = 'o200k_base'  # GPT-4.1 et GPT-4o utilisent o200k_base
        self.encoding = None
        self._initialize_encoding()
    
    def _initialize_encoding(self):
        """Initialise l'encoding de manière robuste"""
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
            st.info(f"✅ Encoding {self.encoding_name} initialisé")
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation de l'encoding : {e}")
            # Fallback sur cl100k_base si o200k_base échoue
            try:
                self.encoding = tiktoken.get_encoding('cl100k_base')
                self.encoding_name = 'cl100k_base'
                st.warning("⚠️ Utilisation de l'encoding de fallback : cl100k_base")
            except Exception as e2:
                st.error(f"❌ Impossible d'initialiser un encoding : {e2}")
                self.encoding = None
    
    def _get_available_model(self):
        """Détermine le meilleur modèle disponible"""
        for model in self.model_priority:
            try:
                # Test rapide pour voir si le modèle est disponible
                test_response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                    timeout=10
                )
                st.success(f"✅ Modèle {model} utilisé avec succès !")
                return model
            except Exception as e:
                st.warning(f"⚠️ Modèle {model} non disponible : {str(e)[:100]}...")
                continue
        
        # Fallback sur gpt-4o-mini par défaut
        st.info("📝 Utilisation du modèle de fallback : gpt-4o-mini")
        return 'gpt-4o-mini'
    
    def analyze_sentence(self, sentence):
        """Analyse complète d'une phrase"""
        cleaned_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        words = cleaned_sentence.split()
        
        analysis = {
            'original': sentence,
            'cleaned': cleaned_sentence,
            'word_count': len(words),
            'char_count': len(sentence),
            'unique_words': len(set(words)),
            'words': words
        }
        
        return analysis
    
    def tokenize_sentence_openai(self, sentence):
        """Tokenisation avec le modèle sélectionné"""
        if not self.encoding:
            return {'error': 'Encoding non initialisé'}
        
        try:
            tokens = self.encoding.encode(sentence)
            
            token_strings = []
            for token in tokens:
                try:
                    token_bytes = self.encoding.decode_single_token_bytes(token)
                    token_string = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_string = f"<bytes:{token_bytes.hex()}>"
                except Exception:
                    # Fallback pour les tokens qui ne peuvent pas être décodés
                    token_string = f"<token_id:{token}>"
                token_strings.append(token_string)
            
            decoded_text = self.encoding.decode(tokens)
            
            return {
                'model': self.model,
                'encoding': self.encoding_name,
                'tokens': tokens,
                'token_strings': token_strings,
                'token_count': len(tokens),
                'decoded_text': decoded_text,
                'is_lossless': decoded_text == sentence
            }
        except Exception as e:
            st.error(f"Erreur de tokenisation : {e}")
            return {'error': str(e)}
    
    def get_important_words_gpt(self, sentence):
        """Analyse de tous les mots de la phrase avec le modèle sélectionné"""
        try:
            # Prompt optimisé pour GPT-4.1
            system_prompt = """Tu es un expert en analyse linguistique. Analyse la phrase et donne un score d'importance (0-1) pour CHAQUE mot de la phrase, y compris les articles, prépositions, etc. 

Tous les mots doivent être inclus dans l'analyse. Retourne uniquement un JSON valide avec ce format exact:
{
    "mots": [
        {"mot": "word1", "score": 0.95},
        {"mot": "word2", "score": 0.85}
    ]
}

Ne pas inclure d'explications, seulement le JSON."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Analyse TOUS les mots de cette phrase : '{sentence}'"
                    }
                ],
                temperature=0,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Nettoyage du contenu pour s'assurer qu'il s'agit d'un JSON valide
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            return result.get('mots', [])
        except json.JSONDecodeError as e:
            st.error(f"Erreur de parsing JSON pour l'analyse : {e}")
            st.error(f"Contenu reçu : {response.choices[0].message.content}")
            return []
        except Exception as e:
            st.error(f"Erreur API OpenAI pour l'analyse : {e}")
            return []
    
    def predict_next_words(self, sentence, num_words=1, top_k=5):
        """Prédiction de plusieurs mots suivants avec le modèle sélectionné"""
        try:
            # Prompt très spécifique pour un seul mot
            system_prompt = f"""Tu es un expert en prédiction de texte utilisant {self.model}. 

TÂCHE CRUCIALE: Prédis EXACTEMENT UN SEUL MOT qui suit logiquement la phrase donnée.

RÈGLES STRICTES:
- UN seul mot par prédiction (pas de phrases ou groupes de mots)
- Même les articles (le, la, les, un, une, des) comptent comme UN mot
- Les contractions (l', d', n') comptent comme UN mot
- Pas d'espaces dans les prédictions
- {top_k} prédictions différentes et uniques
- Probabilités entre 0 et 1

EXEMPLES VALIDES:
- "mange" (verbe)
- "le" (article)  
- "très" (adverbe)
- "l'" (contraction)

EXEMPLES INVALIDES:
- "le chat" (deux mots)
- "très bien" (deux mots)
- "d'accord" (acceptable seulement si c'est un seul token)

FORMAT DE RÉPONSE: Retourne UNIQUEMENT un JSON valide:
{{
    "predictions": [
        {{"sequence": "mot1", "probabilite": 0.85}},
        {{"sequence": "mot2", "probabilite": 0.75}},
        {{"sequence": "mot3", "probabilite": 0.65}},
        {{"sequence": "mot4", "probabilite": 0.55}},
        {{"sequence": "mot5", "probabilite": 0.45}}
    ]
}}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Prédis le mot suivant pour : '{sentence}'"
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # Nettoyage du contenu JSON
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            raw_predictions = result.get('predictions', [])
            
            # Validation stricte pour UN SEUL MOT
            valid_predictions = []
            seen_sequences = set()
            
            for p in raw_predictions:
                if isinstance(p, dict) and 'sequence' in p and 'probabilite' in p:
                    sequence = str(p['sequence']).strip()
                    probability = float(p['probabilite'])
                    
                    # Validation stricte : UN SEUL MOT
                    # Compter les mots en ignorant les apostrophes qui font partie d'un mot
                    words_in_sequence = sequence.split()
                    
                    # Si plusieurs mots, prendre seulement le premier et avertir
                    if len(words_in_sequence) > 1:
                        st.warning(f"Prédiction tronquée de '{sequence}' à '{words_in_sequence[0]}'")
                        sequence = words_in_sequence[0]
                        words_in_sequence = [sequence]
                    
                    is_single_word = len(words_in_sequence) == 1
                    
                    # Vérifier qu'il n'y a pas d'espaces multiples cachés
                    has_multiple_spaces = '  ' in sequence or sequence != sequence.strip()
                    
                    # Validation finale
                    if (sequence and 
                        sequence not in seen_sequences and 
                        0 <= probability <= 1 and 
                        is_single_word and 
                        not has_multiple_spaces and
                        len(sequence) > 0 and
                        sequence.replace("'", "").replace("-", "").isalpha()):  # Vérifier que c'est bien un mot
                        
                        valid_predictions.append((sequence, probability))
                        seen_sequences.add(sequence)
                        
                    else:
                        # Debug : afficher pourquoi la prédiction a été rejetée
                        if len(words_in_sequence) > 1:
                            st.warning(f"Prédiction rejetée (plusieurs mots) : '{sequence}'")
                        elif sequence in seen_sequences:
                            st.warning(f"Prédiction rejetée (doublon) : '{sequence}'")
                        elif not (0 <= probability <= 1):
                            st.warning(f"Prédiction rejetée (probabilité invalide) : '{sequence}' ({probability})")
            
            # Trier par probabilité décroissante et limiter à top_k
            valid_predictions.sort(key=lambda x: x[1], reverse=True)
            final_predictions = valid_predictions[:top_k]
            
            if len(final_predictions) < top_k:
                st.info(f"Seulement {len(final_predictions)} prédictions valides obtenues sur {top_k} demandées")
            
            return final_predictions
            
        except json.JSONDecodeError as e:
            st.error(f"Erreur de parsing JSON pour la prédiction : {e}")
            st.error(f"Contenu reçu : {response.choices[0].message.content}")
            return []
        except Exception as e:
            st.error(f"Erreur API OpenAI pour la prédiction : {e}")
            return []
    
    def generate_continuation(self, sentence, max_length=50, num_sequences=3):
        """Génération de suites logiques avec le modèle sélectionné"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un expert en génération de texte. Continue la phrase donnée de manière logique et cohérente. Génère {num_sequences} continuations différentes d'environ {max_length} mots chacune. Retourne uniquement un JSON avec format: {{\"continuations\": [\"suite1\", \"suite2\", ...]}}"
                    },
                    {
                        "role": "user",
                        "content": f"Continue cette phrase : '{sentence}'"
                    }
                ],
                temperature=0.8
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('continuations', [])
        except Exception as e:
            st.error(f"Erreur API OpenAI pour la génération : {e}")
            return []

    def generate_continuation_from_predictions(self, sentence, predictions, target_word_count=40):
        """Génère 5 textes complets d'environ target_word_count mots en utilisant les 5 mots les plus probables."""
        try:
            if not predictions or not isinstance(predictions, list):
                st.warning("Aucune prédiction disponible pour générer les textes.")
                return []
            
            # Prendre les 5 premières prédictions
            top_5_sequences = [pred[0] for pred in predictions[:5] if len(pred) >= 2]
            
            if not top_5_sequences:
                st.warning("Aucune séquence valide trouvée dans les prédictions.")
                return []
            
            generated_texts = []
            for i, seq in enumerate(top_5_sequences):
                try:
                    first_predicted_word = seq.split()[0] if seq else ""
                    if not first_predicted_word:
                        continue
                        
                    base_for_generation = f"{sentence} {first_predicted_word}"
                    
                    # Prompt optimisé pour GPT-4.1
                    system_prompt = f"""Tu es un assistant d'écriture expert utilisant {self.model}. 

TÂCHE: Continue le texte pour créer un passage d'environ {target_word_count} mots au total.
EXIGENCES: 
- Texte cohérent et naturel
- Se termine proprement avec une ponctuation appropriée
- Style fluide et engageant

RÉPONSE: Retourne uniquement le texte complet (phrase de départ + continuation)."""
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": f"Texte à continuer : '{base_for_generation}'"
                            }
                        ],
                        temperature=0.7,
                        max_tokens=min(300, int(target_word_count * 2))
                    )
                    
                    full_text = response.choices[0].message.content.strip()
                    if full_text:
                        generated_texts.append(full_text)
                        
                except Exception as e:
                    st.warning(f"Erreur lors de la génération du texte {i+1} : {e}")
                    continue
            
            return generated_texts
            
        except Exception as e:
            st.error(f"Erreur lors de la génération des textes étendus : {e}")
            return []

def create_token_visualization(tokenization_result):
    """Crée une visualisation interactive des tokens avec couleurs distinctes"""
    if 'error' in tokenization_result:
        return None
    
    tokens = tokenization_result['token_strings']
    token_ids = tokenization_result['tokens']
    
    import plotly.colors as pc
    
    if len(tokens) <= 10:
        colors = pc.qualitative.Set3[:len(tokens)]
    elif len(tokens) <= 24:
        colors = pc.qualitative.Dark24[:len(tokens)]
    else:
        base_colors = pc.qualitative.Dark24
        colors = [base_colors[i % len(base_colors)] for i in range(len(tokens))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(tokens))),
            y=token_ids,
            text=tokens,
            textposition='auto',
            hovertemplate='<b>Token:</b> %{text}<br><b>ID:</b> %{y}<br><b>Position:</b> %{x}<extra></extra>',
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1
        )
    ])
    
    model_name = tokenization_result.get('model', 'GPT-4.1')
    fig.update_layout(
        title=f"🔍 Visualisation des Tokens {model_name} (Couleurs Distinctes)",
        xaxis_title="Position du Token",
        yaxis_title="ID du Token",
        height=400
    )
    
    return fig

def create_attention_heatmap(important_words):
    """Crée un histogramme des scores d'attention avec gradient de couleur rouge-vert"""
    if not important_words:
        return None
    
    # Agréger les scores pour les mots dupliqués, en gardant le score le plus élevé
    aggregated_scores = {}
    for item in important_words:
        mot = item['mot']
        score = item['score']
        if mot in aggregated_scores:
            aggregated_scores[mot] = max(aggregated_scores[mot], score)
        else:
            aggregated_scores[mot] = score
    
    # Trier les mots par leur score d'attention
    sorted_aggregated_scores = dict(sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True))
    
    words = list(sorted_aggregated_scores.keys())
    scores = list(sorted_aggregated_scores.values())
    
    if not words:
        return None
    
    colors = []
    for score in scores:
        red = int(255 * score)
        green = int(255 * (1 - score))
        blue = 0
        colors.append(f'rgb({red},{green},{blue})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=words,
            y=scores,
            marker_color=colors,
            text=[f'{score:.3f}' for score in scores],
            textposition='auto',
            hovertemplate='<b>Mot:</b> %{x}<br><b>Score d\'Attention:</b> %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="🎯 Scores d'Attention GPT-4.1 (Rouge = Élevé, Vert = Faible)",
        xaxis_title="Mots",
        yaxis_title="Score d'Attention",
        height=400,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_prediction_histogram(predictions_list, num_words_display):
    """Crée un histogramme Plotly des prédictions de mots avec un gradient de couleur"""
    if not predictions_list:
        return go.Figure().update_layout(title="Aucune prédiction à afficher")
    
    if isinstance(predictions_list, dict) and 'error' in predictions_list:
        return go.Figure().update_layout(title=f"Erreur: {predictions_list['error']}")
    
    try:
        sorted_predictions = sorted(predictions_list, key=lambda x: x[1], reverse=True)
    except (IndexError, TypeError) as e:
        return go.Figure().update_layout(title=f"Format de prédictions incorrect: {e}")
    
    sequences_all = [item[0] for item in sorted_predictions]
    probs_all = [item[1] for item in sorted_predictions]
    
    if not sequences_all or not probs_all:
        return go.Figure().update_layout(title="Aucune prédiction valide à afficher")
    
    sequences_display = sequences_all[:num_words_display]
    probs_display = probs_all[:num_words_display]
    
    if not probs_display:
        return go.Figure().update_layout(title="Aucune prédiction à afficher après filtrage")
    
    colors = []
    min_prob_display = min(probs_display) if probs_display else 0
    max_prob_display = max(probs_display) if probs_display else 1
    
    for prob in probs_display:
        if max_prob_display == min_prob_display:
            norm_prob = 0.5
        else:
            norm_prob = (prob - min_prob_display) / (max_prob_display - min_prob_display)
        
        red_val = int(200 * norm_prob + 55 * (1 - norm_prob))
        green_val = int(55 * norm_prob + 200 * (1 - norm_prob))
        blue_val = 50
        
        red_val = max(0, min(255, red_val))
        green_val = max(0, min(255, green_val))
        
        colors.append(f'rgb({red_val},{green_val},{blue_val})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=sequences_display,
            y=probs_display,
            marker_color=colors,
            hovertemplate='<b>Séquence:</b> %{x}<br><b>Probabilité:</b> %{y:.3f}<extra></extra>'
        )
    ])
    
    title_text = f"🎲 Prédictions GPT-4.1 : Top {num_words_display} Mot(s) Suivant(s)"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Séquences Prédites",
        yaxis_title="Probabilité",
        height=400
    )
    
    return fig

def get_token_data_for_table(tokenization_result):
    """Prépare les données des tokens pour un affichage tabulaire"""
    if not tokenization_result or 'error' in tokenization_result or not tokenization_result.get('tokens'):
        return pd.DataFrame()
    
    tokens_ids = tokenization_result['tokens']
    token_strings = tokenization_result['token_strings']
    
    df = pd.DataFrame({
        'ID du Token': tokens_ids,
        'Token (texte)': token_strings,
        'Position': range(1, len(tokens_ids) + 1)
    })
    return df

def create_colored_token_html(tokenization_result):
    """Crée une représentation HTML de la phrase avec des tokens colorés"""
    if not tokenization_result or 'error' in tokenization_result or not tokenization_result.get('token_strings'):
        return ""
    
    token_strings = tokenization_result['token_strings']
    
    if len(token_strings) <= 10:
        colors = px.colors.qualitative.Plotly[:len(token_strings)]
    elif len(token_strings) <= 20:
        colors = px.colors.qualitative.Light24[:len(token_strings)]
    else:
        base_colors = px.colors.qualitative.Dark24
        colors = [base_colors[i % len(base_colors)] for i in range(len(token_strings))]
    
    html_parts = []
    for i, token_str in enumerate(token_strings):
        color = colors[i % len(colors)]
        safe_token_str = html.escape(str(token_str))
        html_parts.append(f'<span style="background-color: {color}; color: black; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block;">{safe_token_str}</span>')
    
    return " ".join(html_parts)

def reset_session_state():
    """Fonction pour réinitialiser les parties pertinentes de st.session_state"""
    keys_to_reset = ['tokenization', 'attention', 'predictions', 'num_words_predicted_for_display', 'generated_texts']
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state:
            del st.session_state[key_to_del]
    st.session_state.input_sentence = ""

def main():
    st.set_page_config(
        page_title="Comprendre les 3 fonctions principales d'un LLM avec GPT-4.1", 
        page_icon="🎓", 
        layout="wide"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🎓 Comprendre les LLM avec GPT-4.1</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Exploration Interactive de la Tokenisation et de l'IA Générative</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("🔑 Configuration")
    
    # Configuration OpenAI mise à jour
    client, api_key = get_openai_client()
    
    if api_key:
        st.sidebar.success("✅ Clé API chargée")
    else:
        api_key = st.sidebar.text_input("Clé API OpenAI :", type="password")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                client.models.list()
                st.sidebar.success("✅ Clé API valide")
            except Exception as e:
                st.sidebar.error(f"❌ Clé API invalide : {e}")
                client = None
    
    if not api_key or not client:
        st.warning("⚠️ Veuillez configurer votre clé API OpenAI valide.")
        return
    
    # Initialisation de l'analyseur
    if 'analyzer' not in st.session_state or st.session_state.get('api_key') != api_key:
        try:
            st.session_state.analyzer = TextAnalyzer(api_key)
            st.session_state.api_key = api_key
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation de l'analyseur : {e}")
            return
    
    analyzer = st.session_state.analyzer
    
    # Affichage du modèle utilisé
    st.sidebar.info(f"🤖 Modèle actuel : **{analyzer.model}**")
    st.sidebar.info(f"🔧 Encoding : **{analyzer.encoding_name}**")
    
    st.markdown("### 📝 Phrase à Analyser")
    if "input_sentence" not in st.session_state:
        st.session_state.input_sentence = "les cerises sont rouges donc je vais les"
    
    sentence = st.text_area(
        "Entrez votre phrase :",
        key="input_sentence", 
        height=100
    )
    
    # Boutons d'action principaux
    col1_main, col2_main, col3_main, col4_main, col5_main = st.columns(5)
    
    with col1_main:
        if st.button("🔍 Tokeniser", use_container_width=True, key="btn_tokenize_main"):
            if st.session_state.input_sentence:
                with st.spinner("Tokenisation en cours..."):
                    st.session_state.tokenization = analyzer.tokenize_sentence_openai(st.session_state.input_sentence)
                if 'attention' in st.session_state: del st.session_state.attention
                if 'predictions' in st.session_state: del st.session_state.predictions
                if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour la tokenisation.")
    
    with col2_main:
        if st.button("🎯 Analyser Attention", use_container_width=True, key="btn_attention_main"):
            if st.session_state.input_sentence:
                if 'tokenization' not in st.session_state or not st.session_state.tokenization or st.session_state.tokenization.get('error'):
                    st.warning("Veuillez d'abord tokeniser une phrase avec succès.")
                else:
                    with st.spinner("Analyse d'attention en cours..."):
                        st.session_state.attention = analyzer.get_important_words_gpt(st.session_state.input_sentence)
                    if 'predictions' in st.session_state: del st.session_state.predictions
                    if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                    st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour l'analyse d'attention.")
    
    with col3_main:
        if st.button("🎲 Prédire Mots", use_container_width=True, key="btn_predict_main"):
            if st.session_state.input_sentence:
                if 'attention' not in st.session_state or not st.session_state.attention:
                    st.warning("Veuillez d'abord analyser l'attention avec succès.")
                else:
                    with st.spinner("Prédiction des mots en cours..."):
                        num_words_to_predict = 1  
                        top_k_predictions = 5     
                        st.session_state.predictions = analyzer.predict_next_words(st.session_state.input_sentence, num_words_to_predict, top_k_predictions)
                        st.session_state.num_words_predicted_for_display = top_k_predictions 
                    if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                    st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour la prédiction.")
    
    with col4_main:
        if st.button("📝 Générer 5 Textes", use_container_width=True, key="btn_generate_texts_main"):
            if st.session_state.input_sentence:
                if 'predictions' not in st.session_state or not st.session_state.predictions:
                    st.warning("Veuillez d'abord prédire les mots avec succès.")
                else:
                    with st.spinner("Génération des textes en cours..."):
                        st.session_state.generated_texts = analyzer.generate_continuation_from_predictions(st.session_state.input_sentence, st.session_state.predictions)
                    st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour générer les textes.")

    with col5_main:
        st.button("🔄 Reset", use_container_width=True, key="btn_reset", on_click=reset_session_state)

    # Affichage des résultats
    if 'tokenization' in st.session_state and st.session_state.tokenization and not st.session_state.tokenization.get('error'):
        st.markdown("---")
        st.markdown("### 🔍 Résultats de Tokenisation")
        col1_tok_disp, col2_tok_disp = st.columns(2)
        with col1_tok_disp:
            st.markdown("#### Représentation Colorée des Tokens")
            token_html = create_colored_token_html(st.session_state.tokenization)
            if token_html:
                st.markdown(token_html, unsafe_allow_html=True)
            else:
                st.info("Impossible de générer la représentation colorée des tokens.")
        with col2_tok_disp:
            st.markdown("#### Tableau Détaillé des Tokens")
            token_df = get_token_data_for_table(st.session_state.tokenization)
            if not token_df.empty:
                st.dataframe(token_df.set_index('Position'))
            else:
                st.info("Aucune donnée de token à afficher dans le tableau.")
        
        if st.button("🎯 Analyser Attention", use_container_width=True, key="btn_attention_ctx_after_tokenize"):
            if st.session_state.input_sentence:
                with st.spinner("Analyse d'attention en cours..."):
                    st.session_state.attention = analyzer.get_important_words_gpt(st.session_state.input_sentence)
                if 'predictions' in st.session_state: del st.session_state.predictions
                if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour l'analyse d'attention.")

    if 'attention' in st.session_state and st.session_state.attention:
        st.markdown("---")
        st.markdown("### 🎯 Analyse d'Attention")
        fig_attention = create_attention_heatmap(st.session_state.attention)
        if fig_attention:
            st.plotly_chart(fig_attention, use_container_width=True)
        else:
            st.error("Impossible de générer l'histogramme d'attention.")
        
        if st.button("🎲 Prédire Mots", use_container_width=True, key="btn_predict_ctx_after_attention"):
            if st.session_state.input_sentence:
                with st.spinner("Prédiction des mots en cours..."):
                    num_words_to_predict = 1  
                    top_k_predictions = 5     
                    st.session_state.predictions = analyzer.predict_next_words(st.session_state.input_sentence, num_words_to_predict, top_k_predictions)
                    st.session_state.num_words_predicted_for_display = top_k_predictions
                if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour la prédiction.")

    if 'predictions' in st.session_state and st.session_state.predictions:
        st.markdown("---")
        num_words_display = st.session_state.get('num_words_predicted_for_display', 5)
        st.markdown(f"### 🎲 Top {num_words_display} Prédictions du Mot Suivant") 
        col_data_pred, col_viz_pred = st.columns([1, 2])
        with col_data_pred:
            if st.session_state.predictions and isinstance(st.session_state.predictions, list):
                display_data_pred = st.session_state.predictions[:num_words_display]
                if display_data_pred:
                    df_pred = pd.DataFrame(display_data_pred, columns=['Séquence', 'Probabilité'])
                    st.dataframe(df_pred, use_container_width=True)
                else:
                    st.info("Aucune donnée de prédiction à afficher dans le tableau.")    
            else:
                st.info("Format de données de prédiction inattendu ou vide.")
        with col_viz_pred:
            fig_pred = create_prediction_histogram(st.session_state.predictions, num_words_display) 
            if fig_pred:
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.error("Impossible de générer le graphique des prédictions.")

        if st.button("📝 Générer 5 Textes", use_container_width=True, key="btn_generate_ctx_after_predict"):
            if st.session_state.input_sentence:
                with st.spinner("Génération des textes en cours..."):
                    st.session_state.generated_texts = analyzer.generate_continuation_from_predictions(st.session_state.input_sentence, st.session_state.predictions)
                st.rerun()
            else:
                st.warning("Veuillez entrer une phrase pour générer les textes.")

    if 'generated_texts' in st.session_state and st.session_state.generated_texts:
        st.markdown("---")
        st.markdown("### 📝 5 Textes Générés à partir des Prédictions")
        for i, text in enumerate(st.session_state.generated_texts, 1):
            st.markdown(f"**Texte {i}:** {text}")
    
    # Informations pédagogiques dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Guide Pédagogique")
    st.sidebar.info(f"""
    **Étapes d'Analyse :**
    
    1. 🔍 **Tokenisation** : Découpage en tokens.
    2. 🎯 **Attention** : Identification des mots importants.
    3. 🎲 **Prédiction** : Génération des mots suivants.
    4. 📝 **Génération** : Création de textes complets.
    
    **Modèle utilisé :** {analyzer.model}
    **Context window :** 1M tokens (GPT-4.1)
    **Knowledge cutoff :** Juin 2024
    
    **Avantages GPT-4.1 :**
    • +21% en codage vs GPT-4o
    • +10% en suivi d'instructions  
    • Meilleur contexte long
    • Moins de latence
    """)

if __name__ == "__main__":
    main()
