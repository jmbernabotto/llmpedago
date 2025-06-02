import streamlit as st
import openai
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
import html # <--- AJOUTER CET IMPORT
import time

# Charger les variables d'environnement
load_dotenv()
# Récupération sécurisée de la clé
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Ou pour les nouvelles versions d'OpenAI :
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class TextAnalyzer:
    def __init__(self, openai_api_key: str):
        # Configuration OpenAI - Focus sur GPT-4o-mini uniquement
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = 'gpt-4o-mini'  # Modèle unique pour la pédagogie
        self.encoding_name = 'o200k_base'
    
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
        """Tokenisation avec GPT-4o-mini uniquement"""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            tokens = encoding.encode(sentence)
            
            token_strings = []
            for token in tokens:
                try:
                    token_bytes = encoding.decode_single_token_bytes(token)
                    token_string = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_string = f"<bytes:{token_bytes.hex()}>"
                token_strings.append(token_string)
            
            decoded_text = encoding.decode(tokens)
            
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
            return {'error': str(e)}
    
    def get_important_words_gpt(self, sentence):
        """Analyse de tous les mots de la phrase avec GPT-4o-mini"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en analyse linguistique. Analyse la phrase et donne un score d'importance (0-1) pour CHAQUE mot de la phrase, y compris les articles, prépositions, etc. Tous les mots doivent être inclus dans l'analyse. Retourne uniquement un JSON avec format: {{\"mots\": [{{\"mot\": \"word\", \"score\": 0.95}}, ...]}}"
                    },
                    {
                        "role": "user",
                        "content": f"Analyse TOUS les mots de cette phrase : '{sentence}'"
                    }
                ],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('mots', [])
        except Exception as e:
            st.error(f"Erreur API OpenAI pour l'analyse : {e}")
            return []
    
    def predict_next_words(self, sentence, num_words=1, top_k=5):
        """Prédiction de plusieurs mots suivants avec GPT-4o-mini"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un expert en prédiction de texte. Prédis les {num_words} mot(s) suivant(s) les plus probables pour compléter la phrase donnée. Donne {top_k} options différentes et uniques avec leur probabilité estimée. Retourne uniquement un JSON avec format: {{{{ 'predictions': [{{{{ 'sequence': 'mot(s) prédit(s)', 'probabilite': 0.85 }}}}, ...] }}}}"
                    },
                    {
                        "role": "user",
                        "content": f"Prédis les {num_words} mot(s) suivant(s) pour : '{sentence}'"
                    }
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            raw_predictions = result.get('predictions', [])
            
            # Post-traitement pour garantir l'unicité et le nombre correct de prédictions
            unique_predictions_dict = {}
            for p in raw_predictions:
                sequence = p['sequence']
                probability = p['probabilite']
                # Si la séquence n'est pas déjà vue, ou si la nouvelle probabilité est meilleure
                if sequence not in unique_predictions_dict or probability > unique_predictions_dict[sequence]:
                    unique_predictions_dict[sequence] = probability
            
            # Trier par probabilité (décroissante) et prendre les top_k
            # Convertir le dictionnaire en une liste de tuples (séquence, probabilité)
            sorted_unique_predictions = sorted(unique_predictions_dict.items(), key=lambda item: item[1], reverse=True)
            
            # Retourner jusqu'à top_k prédictions uniques
            return sorted_unique_predictions[:top_k]
            
        except Exception as e:
            st.error(f"Erreur API OpenAI pour la prédiction : {e}")
            return []
    
    def generate_continuation(self, sentence, max_length=50, num_sequences=3):
        """Génération de suites logiques avec GPT-4o-mini"""
        # Cette fonction n'est plus directement utilisée par un bouton mais peut être conservée pour un usage futur.
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
            
            top_5_sequences = [pred[0] for pred in predictions[:5]]
            
            generated_texts = []
            for seq in top_5_sequences:
                if seq:
                    first_predicted_word = seq.split()[0]
                    base_for_generation = f"{sentence} {first_predicted_word}"
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": f"Tu es un assistant d'écriture. À partir de la phrase fournie : '{base_for_generation}', continue d'écrire pour produire un texte qui, au total (phrase de départ incluse), fait environ {target_word_count} mots. Le texte doit être cohérent, naturel et se terminer par un point. Retourne le texte complet (phrase de départ + continuation)."
                            },
                            {
                                "role": "user",
                                "content": f"Phrase de départ : '{base_for_generation}'. Continue à écrire."
                            }
                        ],
                        temperature=0.7,
                        max_tokens=int(target_word_count * 1.5) 
                    )
                    
                    # La réponse du modèle devrait maintenant être le texte complet.
                    full_text = response.choices[0].message.content.strip()
                    generated_texts.append(full_text)
            
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
    
    fig.update_layout(
        title="🔍 Visualisation des Tokens GPT-4o-mini (Couleurs Distinctes)",
        xaxis_title="Position du Token",
        yaxis_title="ID du Token",
        height=400
    )
    
    return fig

def create_attention_heatmap(important_words):
    """Crée un histogramme des scores d'attention avec gradient de couleur rouge-vert,
       en s'assurant que chaque mot n'apparaît qu'une fois avec son score le plus élevé."""
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
            
    # Trier les mots par leur score d'attention (facultatif, mais peut améliorer la lisibilité)
    # Trié du plus important au moins important
    sorted_aggregated_scores = dict(sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True))

    words = list(sorted_aggregated_scores.keys())
    scores = list(sorted_aggregated_scores.values())
    
    if not words: # Vérifier si après agrégation, il reste des mots
        return None

    colors = []
    # Les scores sont maintenant entre 0 et 1, normalisés par le modèle GPT.
    # Le gradient ira du rouge (score proche de 1) au vert (score proche de 0).
    for score in scores:
        # Rouge intense pour score élevé, Vert intense pour score faible
        red = int(255 * score)        # Plus le score est élevé, plus il y a de rouge
        green = int(255 * (1 - score)) # Plus le score est bas, plus il y a de vert
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
        title="🎯 Histogramme des Scores d'Attention (Rouge = Élevé, Vert = Faible)",
        xaxis_title="Mots",
        yaxis_title="Score d'Attention",
        height=400,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_prediction_histogram(predictions_list, num_words_display):
    """Crée un histogramme Plotly des prédictions de mots avec un gradient de couleur.
    predictions_list: une liste de tuples/listes, ex: [('seq1', 0.8), ('seq2', 0.7), ...]
    """
    if not predictions_list:
        return go.Figure().update_layout(title="Aucune prédiction à afficher")
    
    # Si predictions_list est un dict avec une clé 'error', c'est une erreur
    if isinstance(predictions_list, dict) and 'error' in predictions_list:
        return go.Figure().update_layout(title=f"Erreur: {predictions_list['error']}")

    # Extraire les séquences et leurs probabilités de la liste
    # La liste est supposée être déjà triée par probabilité par predict_next_words
    # ou nous pouvons la trier ici si nécessaire.
    # Pour l'instant, supposons qu'elle est dans l'ordre souhaité ou que l'ordre n'importe pas avant le tri interne.
    
    # Assurons-nous que les éléments sont des paires (séquence, probabilité)
    try:
        # Trier par probabilité (deuxième élément de la paire), du plus haut au plus bas
        sorted_predictions = sorted(predictions_list, key=lambda x: x[1], reverse=True)
    except (IndexError, TypeError) as e:
        return go.Figure().update_layout(title=f"Format de prédictions incorrect: {e}")

    sequences_all = [item[0] for item in sorted_predictions]
    probs_all = [item[1] for item in sorted_predictions]

    if not sequences_all or not probs_all:
        return go.Figure().update_layout(title="Aucune prédiction valide à afficher")

    # Limiter au nombre de mots à afficher
    sequences_display = sequences_all[:num_words_display]
    probs_display = probs_all[:num_words_display]

    if not probs_display:
        return go.Figure().update_layout(title="Aucune prédiction à afficher après filtrage")

    # Générer les couleurs avec un gradient de rouge (plus probable) à vert (moins probable)
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
    
    # Le nombre de mots prédits est implicitement 1 par prédiction individuelle ici
    # Si num_words_predicted était dans predictions_list, il faudrait l'extraire.
    # Pour l'instant, on se base sur num_words_display pour le titre.
    title_text = f"🎲 Prédictions des {num_words_display} Mot(s) Suivant(s)" if num_words_display > 1 else "🎲 Prédictions du Mot Suivant"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Séquences Prédites",
        yaxis_title="Probabilité",
        height=400
    )
    
    return fig

def get_token_data_for_table(tokenization_result):
    """Prépare les données des tokens pour un affichage tabulaire."""
    if not tokenization_result or 'error' in tokenization_result or not tokenization_result.get('tokens'):
        return pd.DataFrame()

    tokens_ids = tokenization_result['tokens']
    token_strings = tokenization_result['token_strings']
    
    # Crée un DataFrame pour une meilleure lisibilité
    df = pd.DataFrame({
        'ID du Token': tokens_ids,
        'Token (texte)': token_strings,
        'Position': range(1, len(tokens_ids) + 1)
    })
    return df

def create_colored_token_html(tokenization_result):
    """Crée une représentation HTML de la phrase avec des tokens colorés."""
    if not tokenization_result or 'error' in tokenization_result or not tokenization_result.get('token_strings'):
        return ""

    token_strings = tokenization_result['token_strings']
    
    # Générer une palette de couleurs distinctes
    if len(token_strings) <= 10:
        colors = px.colors.qualitative.Plotly[:len(token_strings)] # <--- MODIFIER ICI (pc -> px)
    elif len(token_strings) <= 20:
        colors = px.colors.qualitative.Light24[:len(token_strings)] # <--- MODIFIER ICI (pc -> px)
    else: # Pour plus de 20 tokens, on cycle sur une palette plus large
        base_colors = px.colors.qualitative.Dark24 # <--- MODIFIER ICI (pc -> px)
        colors = [base_colors[i % len(base_colors)] for i in range(len(token_strings))]

    html_parts = []
    for i, token_str in enumerate(token_strings):
        color = colors[i % len(colors)] # Cycle à travers les couleurs si plus de tokens que de couleurs
        # Échapper les caractères HTML spéciaux dans le token avant de l'insérer
        safe_token_str = html.escape(str(token_str)) # <--- MODIFIER CETTE LIGNE (ajout de str() pour s'assurer que c'est une chaîne)
        html_parts.append(f'<span style="background-color: {color}; color: black; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block;">{safe_token_str}</span>')
    
    return " ".join(html_parts)

def reset_session_state():
    """Fonction pour réinitialiser les parties pertinentes de st.session_state."""
    keys_to_reset = ['tokenization', 'attention', 'predictions', 'num_words_predicted_for_display', 'generated_texts']
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state:
            del st.session_state[key_to_del]
    # Réinitialiser le champ de texte
    st.session_state.input_sentence = ""

def main():
    st.set_page_config(
        page_title="Comprendre les 3 fonctions principales d'un LLM", 
        page_icon="🎓", 
        layout="wide"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🎓 Comprendre les 3 fonctions principales d'un LLM</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Exploration Interactive de la Tokenisation et de l'IA Générative</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("🔑 Configuration")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        st.sidebar.success("✅ Clé API chargée")
    else:
        api_key = st.sidebar.text_input("Clé API OpenAI :", type="password")
    
    if not api_key:
        st.warning("⚠️ Veuillez configurer votre clé API OpenAI.")
        return
    
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = TextAnalyzer(api_key)
            st.success("✅ Analyseur GPT-4o-mini initialisé !")
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation de l'analyseur : {e}")
            return
    
    analyzer = st.session_state.analyzer
    
    st.markdown("### 📝 Phrase à Analyser")
    # Assurer que la clé existe dans session_state pour le contrôle et l'initialisation
    if "input_sentence" not in st.session_state:
        st.session_state.input_sentence = "les cerises sont rouges donc je vais les"

    # La variable 'sentence' récupère la valeur actuelle de st.session_state.input_sentence
    # grâce à la clé. Toute modification par l'utilisateur met à jour st.session_state.input_sentence.
    sentence = st.text_area(
        "Entrez votre phrase :",
        key="input_sentence", 
        height=100
    )
    
    # Boutons d'action principaux (toujours visibles en haut)
    col1_main, col2_main, col3_main, col4_main, col5_main = st.columns(5)
    
    with col1_main:
        if st.button("🔍 Tokeniser", use_container_width=True, key="btn_tokenize_main"):
            if st.session_state.input_sentence:
                with st.spinner("Tokenisation en cours..."):
                    st.session_state.tokenization = analyzer.tokenize_sentence_openai(st.session_state.input_sentence)
                if 'attention' in st.session_state: del st.session_state.attention
                if 'predictions' in st.session_state: del st.session_state.predictions
                if 'generated_texts' in st.session_state: del st.session_state.generated_texts
                st.rerun() # Pour afficher les résultats et le bouton contextuel
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
                        time.sleep(0.5) # Ajout d'un délai de 3 secondes pour le test
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

    # --- Affichage des résultats ET des boutons contextuels --- 

    if 'tokenization' in st.session_state and st.session_state.tokenization and not st.session_state.tokenization.get('error'):
        st.markdown("---")
        st.markdown("### 🔍 Résultats de Tokenisation")
        col1_tok_disp, col2_tok_disp = st.columns(2)
        with col1_tok_disp:
            st.markdown("#### Représentation Textuelle Colorée des Tokens")
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
                    time.sleep(0.5) # Ajout d'un délai de 3 secondes pour le test (pour le bouton contextuel aussi)
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
    st.sidebar.info("""
    **Étapes d'Analyse :**
    
    1. 🔍 **Tokenisation** : Découpage en tokens.
    2. 🎯 **Attention** : Identification des mots importants.
    3. 🎲 **Prédiction** : Génération des mots suivants les plus probables.
    4. 📝 **Génération de Textes** : Création de phrases complètes avec les mots prédits.
    
    **Modèle utilisé :** GPT-4o-mini
    **Encoding :** o200k_base
    
    **Échelle d'Attention (Histogramme) :**
    🔴 Rouge foncé = Score d'importance élevé
    🟢 Vert clair = Score d'importance faible
    """)

if __name__ == "__main__":
    main()