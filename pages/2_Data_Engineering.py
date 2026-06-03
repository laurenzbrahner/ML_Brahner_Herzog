import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import arff
import plotly.express as px

st.set_page_config(page_title="Data Engineering", layout="wide")
st.title("Data Engineering")

st.markdown("""
Alle Vorverarbeitungsschritte werden ausschliesslich auf dem Trainingsset berechnet (fit)
und anschliessend auf das Testset uebertragen (transform), um Data Leakage zu vermeiden.
""")

st.markdown("---")

st.subheader("Schritt 1: Fehlende Werte")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
**Vorgehen:**

- Merkmal X37 wird vollstaendig entfernt (44 Prozent fehlend).
  Bei einem derart hohen Fehlanteil fuehrt jede Imputation zu stark verzerrten Werten.

- Merkmal X21 erhaelt einen binaeren Indikator (`X21_missing = 1`),
  der das Fehlen des Wertes als eigenstaendige Information erhaelt.
  Anschliessend wird X21 per Median-Imputation aufgefuellt.

- Alle uebrigen Merkmale mit weniger als 7 Prozent fehlenden Werten
  werden per **Median-Imputation** behandelt. Die Median-Imputation
  ist robuster als die Mittelwert-Imputation bei schiefen Verteilungen.
""")
with col2:
    data = {
        'Merkmal': ['X37', 'X21', 'X27', 'X60', 'X45', 'Restliche (58)'],
        'Fehlend (%)': [43.7, 13.5, 6.4, 5.0, 4.9, '<2'],
        'Massnahme': ['Entfernen', 'Indikator + Median', 'Median', 'Median', 'Median', 'Median']
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Schritt 2: Ausreisser")
st.markdown("""
Ausreisser bei Finanzkennzahlen sind haeufig keine Messfehler, sondern spiegeln reale wirtschaftliche
Extremzustaende wider, die gerade bei insolvenzgefaehrdeten Unternehmen auftreten.
Eine Entfernung dieser Werte wuerde relevante Information vernichten.

Stattdessen wird ein **RobustScaler** eingesetzt, der auf Median und Interquartilsabstand (IQR)
basiert und damit unempfindlich gegenueber extremen Werten ist. Im Gegensatz zum StandardScaler
wird der Wertebereich nicht durch Ausreisser verzerrt.
""")

col1, col2, col3 = st.columns(3)
col1.info("StandardScaler: skaliert auf Mittelwert und Standardabweichung. Anfaellig fuer Ausreisser.")
col2.success("RobustScaler: skaliert auf Median und IQR. Robust gegenueber Ausreissern.")
col3.warning("Entfernen: wuerde wirtschaftlich relevante Extremwerte vernichten.")

st.markdown("---")
st.subheader("Schritt 3: Klassenimbalance")
st.markdown("""
Nur etwa 5 Prozent der Unternehmen im Datensatz sind tatsaechlich bankrott gegangen.
Ein Modell das naiv immer die Mehrheitsklasse vorhersagt erreich eine Accuracy von 95 Prozent,
ist aber wertlos fuer die Insolvenzvorhersage.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Gewaehlte Strategie: Doppelte Gewichtung fuer XGBoost**

1. `scale_pos_weight`: Setzt das Verhaeltnis von negativen zu positiven Klassen
   als internen Gewichtungsfaktor im Verlustfunktionsterm.

2. `sample_weight` im `fit()`-Aufruf: Gewichtet jeden einzelnen Trainingspunkt
   entsprechend seiner Klassenhaeufigkeit.

Die Kombination beider Parameter fuehrt zu einem hoeheren Recall bei der Bankrott-Klasse.
Logistische Regression und Random Forest verwenden `class_weight='balanced'`.
""")

with col2:
    dist_data = pd.DataFrame({
        'Jahr': [1, 2, 3, 4, 5],
        'Horizont': ['5 Jahre', '4 Jahre', '3 Jahre', '2 Jahre', '1 Jahr'],
        'Nicht bankrott': [6756, 9773, 10008, 9277, 5500],
        'Bankrott': [271, 400, 495, 515, 410],
    })
    dist_data['Verhaeltnis'] = (dist_data['Nicht bankrott'] / dist_data['Bankrott']).round(1)
    st.dataframe(dist_data[['Horizont', 'Bankrott', 'Nicht bankrott', 'Verhaeltnis']],
                 use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Schritt 4: Feature Selection")
st.markdown("""
Ein korrelationsbasierter Filteransatz (Schwellwert 0.9) wurde evaluiert.
Fuer baumbasierte Modelle (Random Forest, XGBoost) ergaben Experimente, dass die
Vorhersagequalitaet **ohne** Feature Selection hoeher ist.

**Begruendung:** Entscheidungsbaumverfahren waehlen bei jedem Knoten intern das informativste
Merkmal und ignorieren redundante Merkmale dadurch automatisch. Eine vorgelagerte Selektion
entfernt unter Umstaenden Merkmale, die in Kombination mit anderen Merkmalen nuetzlich waeren.

Im vorliegenden Projekt wird daher keine vorgelagerte Feature Selection angewendet.
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Mit Feature Selection (XGBoost, Durchschnitt F1)", "0.570")
with col2:
    st.metric("Ohne Feature Selection (XGBoost, Durchschnitt F1)", "0.671", delta="+0.101")

st.markdown("---")
st.subheader("Datenaufteilung")
st.markdown("""
Da jeder der fuenf Datensaetze einen eigenstaendigen Prognosehorizont repraesentiert,
wird fuer jeden Datensatz ein separates Modell trainiert.

Aufgrund des begrenzten Anteils an Bankrott-Faellen wird auf ein separates Validierungsset verzichtet.
Stattdessen wird ein stratifizierter **85/15-Split** (Train/Test) eingesetzt.
Die Stratifizierung stellt sicher, dass die Klassenverteilung in beiden Teilmengen erhalten bleibt.
""")

split_data = pd.DataFrame({
    'Jahr': [1, 2, 3, 4, 5],
    'Horizont': ['5 Jahre', '4 Jahre', '3 Jahre', '2 Jahre', '1 Jahr'],
    'Gesamt': [7027, 10173, 10503, 9792, 5910],
    'Train (85%)': [5972, 8647, 8927, 8323, 5023],
    'Test (15%)': [1055, 1526, 1576, 1469, 887],
})
st.dataframe(split_data, use_container_width=True, hide_index=True)
