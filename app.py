import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
# Diese Imports sind nötig, damit pickle die Modelle versteht:
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

# -------------------------------------------------------------------
# 1. SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="MS AI System", layout="wide", page_icon="🧠")

st.title("🧠 Neurology AI: Disease Prediction System")
st.markdown("### Kombinierte Vorhersage (Relapse & Progression)")

# -------------------------------------------------------------------
# 2. EINGABE (Sidebar)
# -------------------------------------------------------------------
st.sidebar.header("Patienten-Daten")


def get_user_input():
    # A. Basis
    age = st.sidebar.slider("Alter bei Diagnose", 18, 80, 35)
    sex = st.sidebar.radio("Geschlecht", ["Weiblich", "Männlich"], horizontal=True)

    # B. Klinik
    edss = st.sidebar.slider("Aktueller EDSS Score", 0.0, 10.0, 3.0, step=0.5)
    relapses = st.sidebar.number_input("Anzahl bisheriger Schübe", 0, 50, 1)

    st.sidebar.subheader("Symptome")
    s_spinal = st.sidebar.checkbox("Rückenmark (Spinal)", value=True)
    s_eye = st.sidebar.checkbox("Sehnerv (Eye)", value=False)
    s_brain = st.sidebar.checkbox("Hirnstamm", value=False)
    s_supra = st.sidebar.checkbox("Supratentoriell", value=False)

    # DataFrame bauen (Muss exakt zum Training passen)
    data = {
        'sex_encoded': [1 if sex == "Männlich" else 0],
        'age_at_onset_1': [age],
        'edss_as_evaluated_by_clinician_1': [edss],
        'num_relapses': [relapses],
        'spinal_cord_symptom_1': [1 if s_spinal else 0],
        'brainstem_symptom_1': [1 if s_brain else 0],
        'eye_symptom_1': [1 if s_eye else 0],
        'supratentorial_symptom_1': [1 if s_supra else 0]
    }
    return pd.DataFrame(data)


input_df = get_user_input()

# -------------------------------------------------------------------
# 3. ANALYSE
# -------------------------------------------------------------------
if st.button("Risiko berechnen", type="primary"):

    col1, col2 = st.columns(2)

    # --- LINKER BEREICH: SCHÜBE (COX) ---
    with col1:
        st.subheader("🔥 Akutes Schub-Risiko")
        try:
            with open('relapse_model.pkl', 'rb') as f:
                model_rel = pickle.load(f)

            # Vorhersage (Survival Function)
            surv = model_rel.predict_survival_function(input_df)

            # --- UPDATE: Intelligente Zeit-Suche ---
            # Wir suchen den Index, der am nächsten an "1.0" (1 Jahr) liegt
            # Anstatt stur 'iloc[12]' zu nehmen.
            target_time = 1.0
            idx = (np.abs(surv.index - target_time)).argmin()
            risk_1y = 1 - surv.iloc[idx, 0]

            st.metric("Schub-Risiko (1 Jahr)", f"{risk_1y:.1%}")

            # Grafik
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(surv.index, surv.values, color="orange", linewidth=2)
            ax.set_title("Wahrscheinlichkeit: Schubfrei")
            # Linie bei 1 Jahr einzeichnen
            ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        except FileNotFoundError:
            st.error("Datei 'relapse_model.pkl' fehlt!")
        except Exception as e:
            st.error(f"Fehler bei Relapse: {e}")

    # --- RECHTER BEREICH: PROGRESSION (RANDOM FOREST) ---
    with col2:
        st.subheader("📉 Langzeit-Progression")
        try:
            with open('progression_forest_model.pkl', 'rb') as f:
                model_rf = pickle.load(f)

            # Vorhersage
            surv_funcs = model_rf.predict_survival_function(input_df)
            curve = surv_funcs[0]

            # Wert bei Jahr 2 suchen (Intelligente Suche war hier schon drin)
            idx = (np.abs(curve.x - 2.0)).argmin()
            risk_2y = 1 - curve.y[idx]

            st.metric("Progressions-Risiko (2 Jahre)", f"{risk_2y:.1%}")

            # Grafik
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.step(curve.x, curve.y, where="post", color="navy", linewidth=2)
            ax2.set_title("Wahrscheinlichkeit: Stabil")
            # Linie bei 2 Jahren einzeichnen
            ax2.axvline(x=2, color="red", linestyle="--", alpha=0.5)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        except FileNotFoundError:
            st.error("Datei 'progression_forest_model.pkl' fehlt!")
        except Exception as e:
            st.error(f"Fehler bei Progression: {e}")


# im terminal ausführen:
 # pip install streamlit scikit-survival lifelines pandas matplotlib
 # streamlit run app.py