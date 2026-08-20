import os
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Telemetry Engineer AI", layout="wide")
st.title("🏎️ Track Telemetry AI Assistant")

# Clé API récupérée depuis les secrets de Streamlit
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not api_key:
    st.error("⚠️ Clé API Gemini manquante. Ajoute GEMINI_API_KEY dans les Secrets Streamlit.")
    st.stop()

client = genai.Client(api_key=api_key)

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

    # Construction du prompt selon la présence de CSV
    if data_summary:
        prompt_content = f"Données de télémétrie actuelles :\n{data_summary}\n\nQuestion du pilote : {user_prompt}"
    else:
        prompt_content = user_prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyse Gemini en cours..."):
            try:
                # Utilisation du modèle gemini-2.0-flash
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                    )
                )
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                # Fallback sur gemini-1.5-flash si le v2 n'est pas disponible sur ta clé
                try:
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            temperature=0.2,
                        )
                    )
                    st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except Exception as err:
                    st.error(f"Erreur d'analyse : {err}")
