import streamlit as st

st.set_page_config(
    page_title="Info",
    page_icon=None,
    layout="wide",
)

st.title("Insolvenzvorhersage polnischer Unternehmen")
st.markdown(
    "### Machine-Learning-Klassifikation auf Basis von Jahresabschlussdaten (2000 bis 2013)")

st.markdown("---")

st.markdown("""
Diese Anwendung praesentiert eine vollstaendige Machine-Learning-Pipeline zur Vorhersage
von Unternehmensinsolvenzen auf Basis des polnischen EMIS-Datensatzes von Tomczak (2016).

Die Navigation befindet sich in der Seitenleiste. Die Seiten bauen aufeinander auf:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**1. Explorative Datenanalyse**

Ueberblick ueber den Datensatz, Klassenverteilung, fehlende Werte
und Merkmalsbedeutsamkeit.

**2. Data Engineering**

Dokumentation aller Vorverarbeitungsschritte: Imputation,
Skalierung und Behandlung der Klassenimbalance.
""")

with col2:
    st.markdown("""
**3. Ergebnisse**

Vergleich aller drei Modelle (Logistische Regression, Random Forest,
XGBoost) über alle Prognosehorizonte auf dem Testset.

**4. Vorhersage**

Interaktive Eingabe von Finanzkennzahlen und direkte Vorhersage
des Insolvenzrisikos fuer ein konkretes Unternehmen.

""")

st.markdown("---")

st.markdown("""
**Datensatz:** Tomczak (2016).
Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction.
*Expert Systems with Applications*, 58, 93-101.

**Modell:** XGBoost mit doppelter Imbalance-Behandlung (scale_pos_weight + sample_weight),
85/15-Train-Test-Split, Median-Imputation, RobustScaler.
""")
