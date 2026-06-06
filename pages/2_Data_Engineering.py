import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import arff
import plotly.express as px

st.set_page_config(page_title="Data Engineering", layout="wide")
st.title("Data Engineering")

st.markdown("""
Alle Vorverarbeitungsschritte werden ausschließlich auf dem Trainingsset berechnet (fit)
und anschließend auf das Testset übertragen (transform), um Data Leakage zu vermeiden.
""")

st.markdown("---")

st.subheader("Schritt 1: Fehlende Werte")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
**Vorgehen**

- Merkmal X37 wird vollständig entfernt (44 Prozent fehlend).
  Bei einem derart hohen Fehlanteil führt jede Imputation zu stark verzerrten Werten.

- Merkmal X21 erhält einen binären Indikator (`X21_missing = 1`),
  der das Fehlen des Wertes als eigenständige Information erhält.
  Anschließend wird X21 per Median-Imputation aufgefüllt.

- Alle übrigen Merkmale mit weniger als 7 Prozent fehlenden Werten
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
st.subheader("Schritt 2: Ausreißer")
st.markdown("""
Ausreißer bei Finanzkennzahlen sind häufig keine Messfehler, sondern spiegeln reale wirtschaftliche
Extremzustände wider, die gerade bei insolvenzgefährdeten Unternehmen auftreten.
Eine Entfernung dieser Werte würde relevante Information vernichten.

Stattdessen wird ein **RobustScaler** eingesetzt, der auf Median und Interquartilsabstand (IQR)
basiert und damit unempfindlich gegenüber extremen Werten ist. Im Gegensatz zum StandardScaler
wird der Wertebereich nicht durch Ausreißer verzerrt.
""")

col1, col2, col3 = st.columns(3)
col1.info("StandardScaler skaliert auf Mittelwert und Standardabweichung. Anfällig für Ausreißer.")
col2.success("RobustScaler skaliert auf Median und IQR. Robust gegenüber Ausreißern.")
col3.warning("Entfernen würde wirtschaftlich relevante Extremwerte vernichten.")

st.markdown("---")
st.subheader("Schritt 3: Klassenimbalance")
st.markdown("""
Nur etwa 5 Prozent der Unternehmen im Datensatz sind tatsächlich bankrott gegangen.
Ein Modell das naiv immer die Mehrheitsklasse vorhersagt erreicht eine Accuracy von 95 Prozent,
ist aber wertlos für die Insolvenzvorhersage.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Gewählte Strategie: Doppelte Gewichtung für XGBoost**

1. `scale_pos_weight` setzt das Verhältnis von negativen zu positiven Klassen
   als internen Gewichtungsfaktor im Verlustfunktionsterm.

2. `sample_weight` im `fit()`-Aufruf gewichtet jeden einzelnen Trainingspunkt
   entsprechend seiner Klassenhäufigkeit.

Die Kombination beider Parameter führt zu einem höheren Recall bei der Bankrott-Klasse.
Logistische Regression und Random Forest verwenden `class_weight='balanced'`.
""")

with col2:
    dist_data = pd.DataFrame({
        'Jahr': [1, 2, 3, 4, 5],
        'Horizont': ['5 Jahre', '4 Jahre', '3 Jahre', '2 Jahre', '1 Jahr'],
        'Nicht bankrott': [6756, 9773, 10008, 9277, 5500],
        'Bankrott': [271, 400, 495, 515, 410],
    })
    dist_data['Verhältnis'] = (dist_data['Nicht bankrott'] / dist_data['Bankrott']).round(1)
    st.dataframe(dist_data[['Horizont', 'Bankrott', 'Nicht bankrott', 'Verhältnis']],
                 use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Schritt 4: Feature Selection")
st.markdown("""
Ein korrelationsbasierter Filteransatz (Schwellwert 0.9) wurde evaluiert.
Für baumbasierte Modelle (Random Forest, XGBoost) ergaben Experimente, dass die
Vorhersagequalität **ohne** Feature Selection höher ist.

**Begründung**

Entscheidungsbaumverfahren wählen bei jedem Knoten intern das informativste
Merkmal und ignorieren redundante Merkmale dadurch automatisch. Eine vorgelagerte Selektion
entfernt unter Umständen Merkmale, die in Kombination mit anderen Merkmalen nützlich wären.

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
Da jeder der fünf Datensätze einen eigenständigen Prognosehorizont repräsentiert,
wird für jeden Datensatz ein separates Modell trainiert.

Aufgrund des begrenzten Anteils an Bankrott-Fällen wird auf ein separates Validierungsset verzichtet.
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
