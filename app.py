import os
import pandas as pd
import streamlit as st
import plotly.express as px
from google import genai
from google.genai import types

st.set_page_config(page_title="Telemetry Engineer AI", layout="wide")

# Récupération de la clé API
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error("⚠️ Clé API Gemini manquante. Ajoute GEMINI_API_KEY dans les Secrets Streamlit.")
    st.stop()

client = genai.Client(api_key=api_key)

def load_aim_csv(file):
    """Lit un fichier CSV AiM Race Studio en sautant les métadonnées de l'en-tête."""
    # Détection automatique de la ligne où commencent les vraies données/colonnes
    file.seek(0)
    lines = [file.readline().decode('utf-8', errors='ignore') for _ in range(50)]
    
    header_line_idx = 0
    for idx, line in enumerate(lines):
        # Recherche d'une ligne contenant des mots-clés typiques de télémétrie ou GPS
        if any(keyword in line.lower() for keyword in ['time', 'lat', 'speed', 'vitesse', 'gps', 'dist']):
            header_line_idx = idx
            break

    file.seek(0)
    # Essai de lecture avec virgule ou point-virgule
    try:
        df = pd.read_csv(file, skiprows=header_line_idx)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, skiprows=header_line_idx, sep=';')
    return df

# ---------------------------------------------------------
# Sidebar : Téléversement
# ---------------------------------------------------------
st.sidebar.header("Données de télémétrie")
uploaded_files = st.sidebar.file_uploader(
    "Téléverse tes CSV (Télémétrie / Secteurs)",
    type=["csv"],
    accept_multiple_files=True
)

data_summary = ""
gps_df = None

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} fichier(s) chargé(s)")
    summaries = []
    for file in uploaded_files:
        try:
            df = load_aim_csv(file)
            
            # Recherche des colonnes GPS AiM (ex: GPS_Lat, GPS_Lon, Latitude, Longitude, etc.)
            lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()), None)
            
            if lat_col and lon_col and gps_df is None:
                # Nettoyage des coordonnées GPS si présentes
                temp_gps = df[[lat_col, lon_col]].dropna()
                temp_gps.columns = ['lat', 'lon']
                # Filtrage basique pour s'assurer que ce ne sont pas des zéros
                gps_df = temp_gps[(temp_gps['lat'] != 0) & (temp_gps['lon'] != 0)]
            
            summary = f"\n--- Fichier {file.name} ---\nColonnes: {list(df.columns)}\n"
            summary += df.describe(include='all').to_string()
            summaries.append(summary)
        except Exception as e:
            st.error(f"Erreur lors de la lecture de {file.name}: {e}")
    data_summary = "\n".join(summaries)

# ---------------------------------------------------------
# En-tête : Titre (5/6) + Carte GPS (1/6)
# ---------------------------------------------------------
col_title, col_gps = st.columns([5, 1])

with col_title:
    st.title("🏎️ Track Telemetry AI Assistant")
    st.caption("Analyse de télémétrie & Data Coaching")

with col_gps:
    if gps_df is not None and not gps_df.empty:
        fig = px.line_mapbox(
            gps_df, 
            lat="lat", 
            lon="lon", 
            zoom=13, 
            height=180
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0, "t":0, "l":0, "b":0}
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("📍 Trace GPS")

st.divider()

# ---------------------------------------------------------
# Section Analyse & Chat
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """
Tu es un Ingénieur Télémétrie et Data Coach expert en sports mécaniques.
Tu réponds au pilote de manière concise, technique et pertinente.
Si des fichiers CSV de télémétrie sont fournis, tu les analyses en détail.
Si aucun fichier n'est téléversé, tu échanges normalement avec le pilote sur les réglages, le pilotage ou la télémétrie.
"""

st.subheader("Analyse & Discussion")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Pose ta question sur tes données de télémétrie ou ton pilotage..."):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    prompt_content = f"Données de télémétrie actuelles :\n{data_summary}\n\nQuestion du pilote : {user_prompt}" if data_summary else user_prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyse Gemini en cours..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                    )
                )
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Erreur lors de l'appel API : {e}")
