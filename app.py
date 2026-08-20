import os
import pandas as pd
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Telemetry Engineer AI", layout="wide")
st.title("🏎️ Track Telemetry AI Assistant")

# Clé API récupérée depuis les secrets de Streamlit
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not api_key:
    st.error("⚠️ Clé API Gemini manquante. Ajoute GEMINI_API_KEY dans les Secrets Streamlit.")
    st.stop()

# Configuration de la clé
genai.configure(api_key=api_key)

st.sidebar.header("Données de télémétrie")
uploaded_files = st.sidebar.file_uploader(
    "Téléverse tes CSV (Télémétrie / Secteurs)",
    type=["csv"],
    accept_multiple_files=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

data_summary = ""
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} fichier(s) chargé(s)")
    summaries = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            st.write(f"**Aperçu : {file.name}**")
            st.dataframe(df.head(3))
            
            summary = f"\n--- Fichier {file.name} ---\nColonnes: {list(df.columns)}\n"
            summary += df.describe(include='all').to_string()
            summaries.append(summary)
        except Exception as e:
            st.error(f"Erreur lors de la lecture de {file.name}: {e}")
    data_summary = "\n".join(summaries)

SYSTEM_INSTRUCTION = """
Tu es un Ingénieur Télémétrie et Data Coach expert en sports mécaniques.
Tu réponds au pilote de manière concise, technique et pertinente.
Si des fichiers CSV de télémétrie sont fournis, tu les analyses en détail.
Si aucun fichier n'est téléversé, tu échanges normalement avec le pilote sur les réglages, le pilotage ou la télémétrie.
"""

st.subheader("Analyse & Discussion")

# Affichage de l'historique
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if user_prompt := st.chat_input("Pose ta question sur tes données de télémétrie ou ton pilotage..."):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    if data_summary:
        prompt_content = f"Données de télémétrie actuelles :\n{data_summary}\n\nQuestion du pilote : {user_prompt}"
    else:
        prompt_content = user_prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyse Gemini en cours..."):
            # Liste des noms de modèles possibles selon l'activation de l'API
            candidate_models = [
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash",
                "gemini-1.5-pro-latest",
                "gemini-1.5-pro",
                "gemini-pro"
            ]
            
            response_text = None
            last_error = None

            for m_name in candidate_models:
                try:
                    model = genai.GenerativeModel(
                        model_name=m_name,
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                    res = model.generate_content(prompt_content)
                    response_text = res.text
                    break
                except Exception as err:
                    last_error = err
                    continue

            if response_text:
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            else:
                st.error(f"Erreur d'accès aux modèles Gemini : {last_error}")
