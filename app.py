import os
import io
import re
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telemetry Engineer AI",
    page_icon="🏎️",
    layout="wide"
)

# ---------------------------------------------------------
# Gemini
# ---------------------------------------------------------

api_key = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY")
)

if not api_key:
    st.error(
        "Clé API Gemini manquante. "
        "Ajoute GEMINI_API_KEY dans les Secrets Streamlit."
    )
    st.stop()

# Modèle configurable depuis les Secrets.
# Exemple :
# GEMINI_MODEL = "gemini-3.5-flash"
MODEL_NAME = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
)

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Impossible d'initialiser Gemini : {e}")
    st.stop()


# =========================================================
# CONSTANTES / NOMS DE COLONNES
# =========================================================

LATITUDE_NAMES = [
    "gps_lat",
    "gps latitude",
    "latitude",
    "lat",
]

LONGITUDE_NAMES = [
    "gps_lon",
    "gps_longitude",
    "longitude",
    "lon",
    "long",
]

TIME_NAMES = [
    "time",
    "timestamp",
    "sample time",
    "lap time",
]

SPEED_NAMES = [
    "gps speed",
    "gps_speed",
    "speed",
    "velocity",
]

DISTANCE_NAMES = [
    "gps distance",
    "gps_distance",
    "distance",
    "dist",
]

LAP_NAMES = [
    "lap",
    "lap number",
    "lap_number",
    "lap no",
    "lapno",
]

BRAKE_NAMES = [
    "brake",
    "braking",
    "brake pressure",
    "front brake",
    "rear brake",
]

TPS_NAMES = [
    "tps",
    "throttle",
    "throttle position",
    "accelerator",
]

GEAR_NAMES = [
    "gear",
    "ecu gear",
]


# =========================================================
# UTILITAIRES
# =========================================================

def normalize_name(name):
    """Normalise un nom de colonne pour faciliter la recherche."""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).strip().lower()
    ).strip()


def find_column(df, candidates, contains=True):
    """
    Recherche une colonne parmi plusieurs noms possibles.
    Retourne le nom réel de la colonne.
    """

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    # 1. Correspondance exacte
    for candidate in candidates:
        candidate_norm = normalize_name(candidate)

        if candidate_norm in normalized:
            return normalized[candidate_norm]

    # 2. Recherche partielle
    if contains:
        for col in df.columns:
            col_norm = normalize_name(col)

            for candidate in candidates:
                candidate_norm = normalize_name(candidate)

                if candidate_norm in col_norm:
                    return col

    return None


def convert_numeric_columns(df):
    """
    Convertit automatiquement les colonnes qui ressemblent
    à des valeurs numériques.

    Gère notamment les décimales avec virgule.
    """

    result = df.copy()

    for col in result.columns:

        if pd.api.types.is_numeric_dtype(result[col]):
            continue

        series = (
            result[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        # Si au moins 70 % des valeurs sont numériques,
        # on considère la colonne comme numérique.
        if valid_ratio >= 0.70:
            result[col] = converted

    return result


# =========================================================
# LECTURE CSV AIM
# =========================================================

def detect_separator(raw_text):
    """
    Détecte le séparateur CSV.
    """

    sample = raw_text[:10000]

    separators = {
        ",": sample.count(","),
        ";": sample.count(";"),
        "\t": sample.count("\t"),
    }

    return max(
        separators,
        key=separators.get
    )


def find_header_line(raw_text):
    """
    Recherche une ligne ressemblant réellement à un header AiM.
    """

    lines = raw_text.splitlines()

    telemetry_keywords = [
        "time",
        "speed",
        "gps",
        "distance",
        "lap",
        "latitude",
        "longitude",
        "tps",
        "gear",
        "brake",
    ]

    best_index = 0
    best_score = 0

    for idx, line in enumerate(lines[:100]):

        line_lower = line.lower()

        score = 0

        for keyword in telemetry_keywords:
            if keyword in line_lower:
                score += 1

        # Un header contient généralement plusieurs séparateurs
        separator_score = (
            line.count(",")
            + line.count(";")
            + line.count("\t")
        )

        if separator_score >= 2:
            score += 1

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


def load_aim_csv(uploaded_file):
    """
    Lecture robuste d'un fichier CSV AiM Race Studio.
    """

    uploaded_file.seek(0)

    raw_bytes = uploaded_file.read()

    # Décodage robuste
    raw_text = raw_bytes.decode(
        "utf-8-sig",
        errors="replace"
    )

    header_line = find_header_line(raw_text)
    separator = detect_separator(raw_text)

    # Reconstruction du fichier à partir de la ligne header
    lines = raw_text.splitlines()

    clean_text = "\n".join(
        lines[header_line:]
    )

    try:
        df = pd.read_csv(
            io.StringIO(clean_text),
            sep=separator,
            engine="python",
            on_bad_lines="skip"
        )
    except Exception:

        # Deuxième tentative
        df = pd.read_csv(
            io.StringIO(clean_text),
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )

    # Suppression des colonnes totalement vides
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Suppression des espaces dans les noms
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    df = convert_numeric_columns(df)

    return df


# =========================================================
# IDENTIFICATION DES CANAUX
# =========================================================

def detect_channels(df):

    channels = {}

    channels["time"] = find_column(
        df,
        TIME_NAMES
    )

    channels["speed"] = find_column(
        df,
        SPEED_NAMES
    )

    channels["distance"] = find_column(
        df,
        DISTANCE_NAMES
    )

    channels["lap"] = find_column(
        df,
        LAP_NAMES
    )

    channels["brake"] = find_column(
        df,
        BRAKE_NAMES
    )

    channels["tps"] = find_column(
        df,
        TPS_NAMES
    )

    channels["gear"] = find_column(
        df,
        GEAR_NAMES
    )

    channels["latitude"] = find_column(
        df,
        LATITUDE_NAMES
    )

    channels["longitude"] = find_column(
        df,
        LONGITUDE_NAMES
    )

    return channels


# =========================================================
# NETTOYAGE GPS
# =========================================================

def prepare_gps(df, channels):

    lat_col = channels.get("latitude")
    lon_col = channels.get("longitude")

    if not lat_col or not lon_col:
        return None

    gps = df[
        [lat_col, lon_col]
    ].copy()

    gps.columns = [
        "lat",
        "lon"
    ]

    gps["lat"] = pd.to_numeric(
        gps["lat"],
        errors="coerce"
    )

    gps["lon"] = pd.to_numeric(
        gps["lon"],
        errors="coerce"
    )

    gps = gps.dropna()

    gps = gps[
        (gps["lat"] != 0)
        &
        (gps["lon"] != 0)
    ]

    # Élimination des coordonnées manifestement aberrantes
    gps = gps[
        gps["lat"].between(-90, 90)
        &
        gps["lon"].between(-180, 180)
    ]

    return gps


# =========================================================
# DETECTION DES TOURS
# =========================================================

def detect_laps(df, channels):

    lap_col = channels.get("lap")

    if lap_col:

        lap_values = pd.to_numeric(
            df[lap_col],
            errors="coerce"
        )

        valid = lap_values.dropna()

        if len(valid) > 0:

            df = df.copy()

            df["_lap_number"] = lap_values

            return df

    # -----------------------------------------------------
    # Si aucun canal LAP n'est disponible :
    # tentative de détection par reset de distance.
    # -----------------------------------------------------

    distance_col = channels.get("distance")

    if distance_col:

        distance = pd.to_numeric(
            df[distance_col],
            errors="coerce"
        )

        # Différence entre points successifs
        diff = distance.diff()

        # Un gros retour vers une petite valeur indique
        # potentiellement le passage de la ligne.
        reset = (
            diff < -max(
                100,
                distance.max() * 0.3
            )
        )

        lap_number = reset.cumsum() + 1

        df = df.copy()
        df["_lap_number"] = lap_number

        return df

    return None


# =========================================================
# ANALYSE DES TOURS
# =========================================================

def calculate_lap_summary(df, channels):

    if "_lap_number" not in df.columns:
        return pd.DataFrame()

    speed_col = channels.get("speed")

    rows = []

    for lap_number, lap_df in df.groupby(
        "_lap_number",
        dropna=True
    ):

        row = {
            "Lap": lap_number,
            "Samples": len(lap_df),
        }

        # -------------------------------------------------
        # Temps
        # -------------------------------------------------

        time_col = channels.get("time")

        if time_col:

            times = pd.to_numeric(
                lap_df[time_col],
                errors="coerce"
            ).dropna()

            if len(times) >= 2:

                lap_duration = (
                    times.iloc[-1]
                    -
                    times.iloc[0]
                )

                row["Lap Time"] = lap_duration

        # -------------------------------------------------
        # Vitesse
        # -------------------------------------------------

        if speed_col:

            speed = pd.to_numeric(
                lap_df[speed_col],
                errors="coerce"
            ).dropna()

            if len(speed) > 0:

                row["Min Speed"] = speed.min()
                row["Max Speed"] = speed.max()
                row["Average Speed"] = speed.mean()

        rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    # -----------------------------------------------------
    # Nettoyage des temps
    # -----------------------------------------------------

    if "Lap Time" in summary.columns:

        summary = summary[
            summary["Lap Time"] > 0
        ]

    return summary


def find_best_lap(lap_summary):

    if lap_summary.empty:
        return None

    if "Lap Time" not in lap_summary.columns:
        return None

    valid = lap_summary[
        lap_summary["Lap Time"].notna()
        &
        (lap_summary["Lap Time"] > 0)
    ]

    if valid.empty:
        return None

    return valid.loc[
        valid["Lap Time"].idxmin()
    ]["Lap"]


# =========================================================
# COMPARAISON DE TOURS
# =========================================================

def compare_laps(df, channels, lap_a, lap_b):

    if "_lap_number" not in df.columns:
        return None

    a = df[
        df["_lap_number"] == lap_a
    ].copy()

    b = df[
        df["_lap_number"] == lap_b
    ].copy()

    if a.empty or b.empty:
        return None

    result = {}

    # -----------------------------------------------------
    # Temps
    # -----------------------------------------------------

    time_col = channels.get("time")

    if time_col:

        ta = pd.to_numeric(
            a[time_col],
            errors="coerce"
        ).dropna()

        tb = pd.to_numeric(
            b[time_col],
            errors="coerce"
        ).dropna()

        if len(ta) >= 2 and len(tb) >= 2:

            duration_a = ta.iloc[-1] - ta.iloc[0]
            duration_b = tb.iloc[-1] - tb.iloc[0]

            result["lap_a_time"] = duration_a
            result["lap_b_time"] = duration_b
            result["delta"] = duration_b - duration_a

    # -----------------------------------------------------
    # Vitesse
    # -----------------------------------------------------

    speed_col = channels.get("speed")

    if speed_col:

        speed_a = pd.to_numeric(
            a[speed_col],
            errors="coerce"
        ).dropna()

        speed_b = pd.to_numeric(
            b[speed_col],
            errors="coerce"
        ).dropna()

        if len(speed_a) > 0 and len(speed_b) > 0:

            result["lap_a_min_speed"] = speed_a.min()
            result["lap_b_min_speed"] = speed_b.min()

            result["lap_a_max_speed"] = speed_a.max()
            result["lap_b_max_speed"] = speed_b.max()

            result["lap_a_avg_speed"] = speed_a.mean()
            result["lap_b_avg_speed"] = speed_b.mean()

    return result


# =========================================================
# ANALYSE FREINAGE
# =========================================================

def analyze_braking(df, channels, lap_number):

    brake_col = channels.get("brake")
    speed_col = channels.get("speed")
    distance_col = channels.get("distance")

    if not brake_col:
        return pd.DataFrame()

    lap = df[
        df["_lap_number"] == lap_number
    ].copy()

    if lap.empty:
        return pd.DataFrame()

    brake = pd.to_numeric(
        lap[brake_col],
        errors="coerce"
    ).fillna(0)

    # Seuil simple.
    # On détecte les moments où le canal frein devient significatif.
    threshold = max(
        brake.max() * 0.05,
        0.01
    )

    braking = brake > threshold

    # Début de chaque zone de freinage
    starts = (
        braking
        &
        ~braking.shift(
            1,
            fill_value=False
        )
    )

    indices = lap.index[
        starts
    ]

    rows = []

    for idx in indices:

        position = lap.index.get_loc(idx)

        row = {
            "Index": idx
        }

        if distance_col:
            row["Distance"] = lap.loc[
                idx,
                distance_col
            ]

        if speed_col:
            row["Entry Speed"] = lap.loc[
                idx,
                speed_col
            ]

        row["Brake Value"] = lap.loc[
            idx,
            brake_col
        ]

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# ACCELERATION
# =========================================================

def calculate_acceleration(df, channels):

    speed_col = channels.get("speed")
    time_col = channels.get("time")

    if not speed_col or not time_col:
        return None

    speed = pd.to_numeric(
        df[speed_col],
        errors="coerce"
    )

    time = pd.to_numeric(
        df[time_col],
        errors="coerce"
    )

    dt = time.diff()

    acceleration = speed.diff() / dt

    acceleration = acceleration.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return acceleration


# =========================================================
# FORMATAGE DES TEMPS
# =========================================================

def format_lap_time(seconds):

    if seconds is None:
        return "N/A"

    try:
        seconds = float(seconds)
    except:
        return "N/A"

    minutes = int(
        seconds // 60
    )

    remaining = seconds - minutes * 60

    return f"{minutes}:{remaining:06.3f}"


# =========================================================
# CREATION DU RESUME POUR GEMINI
# =========================================================

def build_ai_context(
    file_name,
    df,
    channels,
    lap_summary,
    best_lap,
    selected_lap=None,
    comparison=None
):

    context = []

    context.append(
        f"FICHIER : {file_name}"
    )

    context.append(
        f"Nombre de lignes : {len(df)}"
    )

    context.append(
        f"Colonnes détectées : {list(df.columns)}"
    )

    # -----------------------------------------------------
    # Canaux
    # -----------------------------------------------------

    context.append(
        "\nCANAUX IDENTIFIÉS :"
    )

    for name, column in channels.items():

        if column:
            context.append(
                f"- {name}: {column}"
            )

    # -----------------------------------------------------
    # Tours
    # -----------------------------------------------------

    if not lap_summary.empty:

        context.append(
            "\nRÉSUMÉ DES TOURS :"
        )

        display_summary = lap_summary.copy()

        if "Lap Time" in display_summary.columns:
            display_summary["Lap Time"] = (
                display_summary["Lap Time"]
                .apply(format_lap_time)
            )

        context.append(
            display_summary
            .tail(30)
            .to_string(index=False)
        )

    # -----------------------------------------------------
    # Meilleur tour
    # -----------------------------------------------------

    if best_lap is not None:

        context.append(
            f"\nMEILLEUR TOUR : {best_lap}"
        )

    # -----------------------------------------------------
    # Tour sélectionné
    # -----------------------------------------------------

    if selected_lap is not None:

        context.append(
            f"\nTOUR SÉLECTIONNÉ : {selected_lap}"
        )

        selected = df[
            df["_lap_number"] == selected_lap
        ].copy()

        # On ne transmet pas 100 000 lignes à Gemini.
        # Échantillonnage à maximum 300 points.
        if len(selected) > 300:

            indices = np.linspace(
                0,
                len(selected) - 1,
                300
            ).astype(int)

            selected = selected.iloc[
                indices
            ]

        # On conserve uniquement les canaux utiles
        useful_columns = []

        for key in [
            "time",
            "distance",
            "speed",
            "brake",
            "tps",
            "gear"
        ]:

            col = channels.get(key)

            if col and col in selected.columns:
                useful_columns.append(col)

        if useful_columns:

            context.append(
                "\nÉCHANTILLON DE TÉLÉMÉTRIE DU TOUR :"
            )

            context.append(
                selected[
                    useful_columns
                ].to_string(index=False)
            )

    # -----------------------------------------------------
    # Comparaison
    # -----------------------------------------------------

    if comparison:

        context.append(
            "\nCOMPARAISON DE TOURS :"
        )

        for key, value in comparison.items():

            if "time" in key or key == "delta":
                value = format_lap_time(value)

            context.append(
                f"- {key}: {value}"
            )

    return "\n".join(context)


# =========================================================
# PROMPT SYSTEME
# =========================================================

SYSTEM_INSTRUCTION = """
Tu es un ingénieur télémétrie et Data Coach spécialisé
en moto sur circuit.

Tu analyses les données comme un ingénieur de course.

Tes priorités :

1. Identifier les pertes et gains de temps.
2. Comparer les tours.
3. Analyser freinage, vitesse d'entrée, vitesse minimum
   et accélération.
4. Identifier les zones où le pilote peut progresser.
5. Distinguer les faits mesurés des hypothèses.
6. Ne jamais inventer une donnée absente.
7. Si une information n'est pas disponible, le dire clairement.
8. Privilégier les chiffres plutôt que les impressions.
9. Donner des conseils directement exploitables par le pilote.

Quand tu analyses un tour :

- commence par le constat objectif ;
- indique ensuite la cause probable ;
- termine par une action concrète à essayer.

Tu connais les principes de pilotage moto :
trail braking, point de corde, vitesse de sortie,
transfert de charge, remise des gaz, positionnement,
gestion des rapports et exploitation de la largeur de piste.

Ne prétends jamais connaître la trajectoire exacte
si aucune donnée GPS suffisante ne permet de l'établir.

Réponds en français sauf si le pilote demande une autre langue.
"""


# =========================================================
# INTERFACE
# =========================================================

st.sidebar.header(
    "🏎️ Données de télémétrie"
)

uploaded_files = st.sidebar.file_uploader(
    "Téléverse tes CSV AiM Race Studio",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.divider()

st.sidebar.caption(
    f"Modèle Gemini : {MODEL_NAME}"
)


# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================

all_sessions = {}

gps_df = None

if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} fichier(s) chargé(s)"
    )

    for file in uploaded_files:

        try:

            df = load_aim_csv(file)

            channels = detect_channels(df)

            df_with_laps = detect_laps(
                df,
                channels
            )

            lap_summary = pd.DataFrame()

            best_lap = None

            if df_with_laps is not None:

                df = df_with_laps

                lap_summary = calculate_lap_summary(
                    df,
                    channels
                )

                best_lap = find_best_lap(
                    lap_summary
                )

            gps = prepare_gps(
                df,
                channels
            )

            if gps is not None and not gps.empty:

                if gps_df is None:
                    gps_df = gps

            all_sessions[file.name] = {
                "df": df,
                "channels": channels,
                "lap_summary": lap_summary,
                "best_lap": best_lap,
            }

        except Exception as e:

            st.error(
                f"Erreur lors de la lecture de "
                f"{file.name}: {e}"
            )


# =========================================================
# TITRE
# =========================================================

st.title(
    "🏎️ Track Telemetry AI Assistant"
)

st.caption(
    "Analyse de télémétrie AiM • Comparaison de tours • Data Coaching"
)


# =========================================================
# GPS
# =========================================================

if gps_df is not None and not gps_df.empty:

    with st.expander(
        "📍 Trace GPS",
        expanded=False
    ):

        fig = px.line_map(
            gps_df,
            lat="lat",
            lon="lon",
            zoom=13,
            height=500
        )

        fig.update_layout(
            margin={
                "r": 0,
                "t": 0,
                "l": 0,
                "b": 0
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


st.divider()


# =========================================================
# SELECTION DE LA SESSION
# =========================================================

if all_sessions:

    file_names = list(
        all_sessions.keys()
    )

    selected_file = st.selectbox(
        "Session / fichier",
        file_names
    )

    session = all_sessions[
        selected_file
    ]

    df = session["df"]
    channels = session["channels"]
    lap_summary = session["lap_summary"]
    best_lap = session["best_lap"]

else:

    df = None
    channels = {}
    lap_summary = pd.DataFrame()
    best_lap = None


# =========================================================
# DASHBOARD
# =========================================================

if df is not None:

    st.subheader(
        "📊 Analyse de la session"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Points de télémétrie",
        f"{len(df):,}"
    )

    if not lap_summary.empty:

        c2.metric(
            "Tours détectés",
            len(lap_summary)
        )

    else:

        c2.metric(
            "Tours détectés",
            "N/A"
        )

    if best_lap is not None:

        best_row = lap_summary[
            lap_summary["Lap"] == best_lap
        ].iloc[0]

        c3.metric(
            "Meilleur tour",
            format_lap_time(
                best_row.get(
                    "Lap Time"
                )
            )
        )

        c4.metric(
            "Tour",
            str(best_lap)
        )

    else:

        c3.metric(
            "Meilleur tour",
            "N/A"
        )

        c4.metric(
            "Tour",
            "N/A"
        )


# =========================================================
# TABLEAU DES TOURS
# =========================================================

if not lap_summary.empty:

    st.subheader(
        "🏁 Tours"
    )

    display_laps = lap_summary.copy()

    if "Lap Time" in display_laps.columns:

        display_laps["Lap Time"] = (
            display_laps["Lap Time"]
            .apply(format_lap_time)
        )

    st.dataframe(
        display_laps,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SELECTION DES TOURS
# =========================================================

selected_lap = None
comparison_lap = None

if not lap_summary.empty:

    available_laps = (
        lap_summary["Lap"]
        .dropna()
        .tolist()
    )

    st.subheader(
        "🔍 Comparaison"
    )

    col_a, col_b = st.columns(2)

    with col_a:

        default_index = 0

        if best_lap in available_laps:
            default_index = available_laps.index(
                best_lap
            )

        selected_lap = st.selectbox(
            "Tour de référence",
            available_laps,
            index=default_index,
            key="reference_lap"
        )

    with col_b:

        comparison_lap = st.selectbox(
            "Tour à comparer",
            available_laps,
            index=(
                1
                if len(available_laps) > 1
                else 0
            ),
            key="comparison_lap"
        )


# =========================================================
# COMPARAISON
# =========================================================

comparison = None

if (
    df is not None
    and selected_lap is not None
    and comparison_lap is not None
    and selected_lap != comparison_lap
):

    comparison = compare_laps(
        df,
        channels,
        selected_lap,
        comparison_lap
    )

    if comparison:

        st.subheader(
            "⏱️ Différence"
        )

        delta = comparison.get(
            "delta"
        )

        if delta is not None:

            if delta > 0:
                st.warning(
                    f"Tour {comparison_lap} : "
                    f"+{delta:.3f} s"
                )

            else:
                st.success(
                    f"Tour {comparison_lap} : "
                    f"{delta:.3f} s"
                )

        d1, d2, d3 = st.columns(3)

        if "lap_a_min_speed" in comparison:

            d1.metric(
                "Vitesse mini référence",
                f"{comparison['lap_a_min_speed']:.1f}"
            )

        if "lap_b_min_speed" in comparison:

            d2.metric(
                "Vitesse mini comparé",
                f"{comparison['lap_b_min_speed']:.1f}"
            )

        if "lap_a_avg_speed" in comparison:

            d3.metric(
                "Vitesse moyenne référence",
                f"{comparison['lap_a_avg_speed']:.1f}"
            )


# =========================================================
# GRAPHIQUE VITESSE
# =========================================================

if (
    df is not None
    and channels.get("speed")
    and selected_lap is not None
):

    speed_col = channels["speed"]

    plot_df = df[
        df["_lap_number"].isin(
            [
                selected_lap,
                comparison_lap
            ]
        )
    ].copy()

    time_col = channels.get("time")

    if time_col:

        plot_df["Time"] = pd.to_numeric(
            plot_df[time_col],
            errors="coerce"
        )

        plot_df["Speed"] = pd.to_numeric(
            plot_df[speed_col],
            errors="coerce"
        )

        plot_df["Lap"] = (
            plot_df["_lap_number"]
            .astype(str)
        )

        plot_df = plot_df.dropna(
            subset=[
                "Time",
                "Speed"
            ]
        )

        if not plot_df.empty:

            st.subheader(
                "📈 Vitesse"
            )

            fig = px.line(
                plot_df,
                x="Time",
                y="Speed",
                color="Lap",
                labels={
                    "Time": "Temps",
                    "Speed": "Vitesse",
                    "Lap": "Tour"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# ANALYSE FREINAGE
# =========================================================

if (
    df is not None
    and selected_lap is not None
    and channels.get("brake")
):

    braking = analyze_braking(
        df,
        channels,
        selected_lap
    )

    if not braking.empty:

        st.subheader(
            "🛑 Zones de freinage détectées"
        )

        st.dataframe(
            braking,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# CANAUX
# =========================================================

with st.expander(
    "🔧 Canaux détectés"
):

    channel_table = pd.DataFrame(
        [
            {
                "Fonction": key,
                "Colonne AiM": value or "Non détectée"
            }
            for key, value in channels.items()
        ]
    )

    st.dataframe(
        channel_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CHAT
# =========================================================

st.divider()

st.subheader(
    "🤖 Analyse & Discussion"
)

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# Affichage historique
for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# QUESTION
# =========================================================

user_prompt = st.chat_input(
    "Pose ta question sur tes données..."
)

if user_prompt:

    st.chat_message(
        "user"
    ).markdown(
        user_prompt
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # -----------------------------------------------------
    # Construction du contexte
    # -----------------------------------------------------

    if df is not None:

        ai_context = build_ai_context(
            selected_file,
            df,
            channels,
            lap_summary,
            best_lap,
            selected_lap,
            comparison
        )

        final_prompt = f"""
Voici les données actuellement analysées.

{ai_context}

QUESTION DU PILOTE :
{user_prompt}

Analyse les données disponibles.
Ne crée aucune donnée qui n'existe pas.
"""
    else:

        final_prompt = user_prompt


    # -----------------------------------------------------
    # Appel Gemini
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analyse télémétrique..."
        ):

            try:

                # Historique récent
                history_text = ""

                recent_history = (
                    st.session_state.chat_history[-8:]
                )

                for message in recent_history:

                    history_text += (
                        f"\n{message['role'].upper()}:\n"
                        f"{message['content']}\n"
                    )

                complete_prompt = f"""
HISTORIQUE RÉCENT :
{history_text}

CONTEXTE TÉLÉMÉTRIQUE :
{final_prompt}
"""

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=complete_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                        max_output_tokens=3000,
                    )
                )

                answer = response.text

                st.markdown(
                    answer
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = (
                    f"Erreur lors de l'appel Gemini : {e}"
                )

                st.error(
                    error_message
                )


# =========================================================
# RESET CHAT
# =========================================================

st.sidebar.divider()

if st.sidebar.button(
    "🗑️ Effacer la conversation"
):

    st.session_state.chat_history = []

    st.rerun()
