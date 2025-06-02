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
                        "content": f"Tu es un expert en prédiction de texte. Prédis les {num_words} mot(s) suivant(s) les plus probables pour compléter la phrase donnée. Donne {top_k} options différentes avec leur probabilité estimée. Retourne uniquement un JSON avec format: {{\"predictions\": [{{\"sequence\": \"mot(s) prédit(s)\", \"probabilite\": 0.85}}, ...]}}"
                    },
                    {
                        "role": "user",
                        "content": f"Prédis les {num_words} mot(s) suivant(s) pour : '{sentence}'"
                    }
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            predictions = result.get('predictions', [])
            return [(p['sequence'], p['probabilite']) for p in predictions]
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

    def generate_continuation_from_predictions(self, sentence, predictions, target_word_count=30):
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
                                "content": f"Tu es un assistant d'écriture. À partir de la phrase fournie : '{base_for_generation}', continue d'écrire pour produire un texte qui, au total (phrase de départ incluse), fait environ {target_word_count} mots. Le texte doit être cohérent et naturel. Retourne le texte complet (phrase de départ + continuation)."
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
    """Crée un histogramme des scores d'attention avec gradient de couleur rouge-vert"""
    if not important_words:
        return None
    
    words = [w['mot'] for w in important_words]
    scores = [w['score'] for w in important_words]
    
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
        title="🎯 Histogramme des Scores d'Attention (Rouge = Élevé, Vert = Faible)",
        xaxis_title="Mots",
        yaxis_title="Score d'Attention",
        height=400,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_prediction_chart(predictions, num_words_predicted):
    """Crée un graphique des prédictions"""
    if not predictions:
        return None
    
    sequences = [p[0] for p in predictions]
    probs = [p[1] for p in predictions]
    
    max_prob = max(probs) if probs else 1
    min_prob = min(probs) if probs else 0
    
    colors = []
    for prob in probs:
        normalized = (prob - min_prob) / (max_prob - min_prob) if max_prob != min_prob else 0.5
        green_intensity = int(50 + normalized * 150)
        color = f'rgb(0, {green_intensity}, 0)'
        colors.append(color)
    
    fig = go.Figure(data=[
        go.Bar(
            x=sequences,
            y=probs,
            marker_color=colors,
            hovertemplate='<b>Séquence:</b> %{x}<br><b>Probabilité:</b> %{y:.3f}<extra></extra>'
        )
    ])
    
    title = f"🎲 Prédictions des {num_words_predicted} Mot(s) Suivant(s)" if num_words_predicted > 1 else "🎲 Prédictions du Mot Suivant"
    
    fig.update_layout(
        title=title,
        xaxis_title="Séquences Prédites",
        yaxis_title="Probabilité",
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Analyseur Pédagogique GPT-4o-mini", 
        page_icon="🎓", 
        layout="wide"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">🎓 Analyseur Pédagogique GPT-4o-mini</h1>
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
    sentence = st.text_area(
        "Entrez votre phrase :",
        "les cerises sont rouges donc je vais les",
        height=100
    )
    
    # Boutons d'action
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Tokeniser", use_container_width=True, key="btn_tokenize"):
            if sentence:
                st.session_state.tokenization = analyzer.tokenize_sentence_openai(sentence)
            else:
                st.warning("Veuillez entrer une phrase pour la tokenisation.")
    
    with col2:
        if st.button("🎯 Analyser Attention", use_container_width=True, key="btn_attention"):
            if sentence:
                st.session_state.attention = analyzer.get_important_words_gpt(sentence)
            else:
                st.warning("Veuillez entrer une phrase pour l'analyse d'attention.")
    
    with col3:
        if st.button("🎲 Prédire Mots", use_container_width=True, key="btn_predict"):
            if sentence:
                num_words_to_predict = 1  # Valeur fixe
                top_k_predictions = 5     # Valeur fixe
                st.session_state.predictions = analyzer.predict_next_words(sentence, num_words_to_predict, top_k_predictions)
                st.session_state.num_words_predicted_for_display = num_words_to_predict # Pour l'affichage du titre du graphique
            else:
                st.warning("Veuillez entrer une phrase pour la prédiction.")
    
    with col4:
        if st.button("📝 Générer 5 Textes", use_container_width=True, key="btn_generate_texts"):
            if sentence:
                if 'predictions' in st.session_state and st.session_state.predictions:
                    # Appel sans spécifier target_length, utilisera la valeur par défaut de 30
                    st.session_state.generated_texts = analyzer.generate_continuation_from_predictions(sentence, st.session_state.predictions)
                else:
                    st.warning("Veuillez d'abord cliquer sur 'Prédire Mots' pour obtenir des prédictions.")
            else:
                st.warning("Veuillez entrer une phrase pour générer les textes.")

    # Affichage des résultats
    if 'tokenization' in st.session_state and st.session_state.tokenization:
        st.markdown("---")
        st.markdown("### 🔍 Résultats de Tokenisation")
        col_data, col_viz = st.columns([1, 2])
        with col_data:
            st.json(st.session_state.tokenization)
        with col_viz:
            fig = create_token_visualization(st.session_state.tokenization)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Impossible de générer la visualisation des tokens.")

    if 'attention' in st.session_state and st.session_state.attention:
        st.markdown("---")
        st.markdown("### 🎯 Analyse d'Attention")
        fig = create_attention_heatmap(st.session_state.attention)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Impossible de générer l'histogramme d'attention.")

    if 'predictions' in st.session_state and st.session_state.predictions:
        st.markdown("---")
        num_words_display = st.session_state.get('num_words_predicted_for_display', 1)
        st.markdown(f"### 🎲 Prédictions des {num_words_display} Mot(s) Suivant(s) (Top 5)")
        col_data, col_viz = st.columns([1, 2])
        with col_data:
            df = pd.DataFrame(st.session_state.predictions, columns=['Séquence', 'Probabilité'])
            st.dataframe(df, use_container_width=True)
        with col_viz:
            fig = create_prediction_chart(st.session_state.predictions, num_words_display)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Impossible de générer le graphique des prédictions.")

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