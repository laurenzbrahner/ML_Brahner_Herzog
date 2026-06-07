import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ergebnisse", layout="wide")
st.title("Ergebnisse")

st.markdown("""
Drei Klassifikationsverfahren werden eingesetzt und auf dem Testset (15 Prozent) evaluiert.
Die Ergebnisse werden pro Prognosejahr und Modell verglichen.
""")

with st.expander("Hinweis: Warum nicht Accuracy?"):
    st.markdown("""
Bei stark unbalancierten Datensätzen ist **Accuracy** als Metrik ungeeignet. Ein Klassifikator,
der naiv immer die Mehrheitsklasse (nicht bankrott) vorhersagt, erreicht bereits ~95 % Accuracy,
erkennt aber keine einzige Insolvenz. Stattdessen werden folgende Metriken verwendet:

| Metrik | Bedeutung |
|---|---|
| **ROC-AUC** | Schwellenwertunabhängiger Modellvergleich. Misst, wie gut das Modell Bankrott- von Nicht-Bankrott-Fällen trennt. Ein Wert von 1,0 ist perfekt, 0,5 entspricht einem Zufallsmodell. |
| **F1 (Bankrott)** | Harmonisches Mittel aus Precision und Recall für die Bankrott-Klasse. Fasst beide Aspekte in einer Zahl zusammen. |
| **Precision (Bankrott)** | Anteil der korrekt erkannten Insolvenzen an allen vom Modell als Insolvenz eingestuften Fällen. Niedrige Precision bedeutet viele Fehlalarme. |
| **Recall (Bankrott)** | Anteil der tatsächlichen Insolvenzen, die das Modell erkennt. Niedriger Recall bedeutet viele übersehene Insolvenzen. |
""")

all_results = pd.DataFrame([
    {'Jahr': 1, 'Horizont': '5 Jahr(e)', 'Modell': 'Logistic Regression', 'ROC-AUC': 0.7535,
     'F1 (Bankrott)': 0.1503, 'Precision (Bankrott)': 0.0852, 'Recall (Bankrott)': 0.6341},
    {'Jahr': 1, 'Horizont': '5 Jahr(e)', 'Modell': 'Random Forest',      'ROC-AUC': 0.9563,
     'F1 (Bankrott)': 0.2174, 'Precision (Bankrott)': 1.0000, 'Recall (Bankrott)': 0.1220},
    {'Jahr': 1, 'Horizont': '5 Jahr(e)', 'Modell': 'XGBoost',            'ROC-AUC': 0.9740,
     'F1 (Bankrott)': 0.7848, 'Precision (Bankrott)': 0.8158, 'Recall (Bankrott)': 0.7561},
    {'Jahr': 2, 'Horizont': '4 Jahr(e)', 'Modell': 'Logistic Regression', 'ROC-AUC': 0.7001,
     'F1 (Bankrott)': 0.1322, 'Precision (Bankrott)': 0.0723, 'Recall (Bankrott)': 0.7667},
    {'Jahr': 2, 'Horizont': '4 Jahr(e)', 'Modell': 'Random Forest',      'ROC-AUC': 0.8531,
     'F1 (Bankrott)': 0.1231, 'Precision (Bankrott)': 0.8000, 'Recall (Bankrott)': 0.0667},
    {'Jahr': 2, 'Horizont': '4 Jahr(e)', 'Modell': 'XGBoost',            'ROC-AUC': 0.8917,
     'F1 (Bankrott)': 0.6346, 'Precision (Bankrott)': 0.7500, 'Recall (Bankrott)': 0.5500},
    {'Jahr': 3, 'Horizont': '3 Jahr(e)', 'Modell': 'Logistic Regression', 'ROC-AUC': 0.8045,
     'F1 (Bankrott)': 0.2218, 'Precision (Bankrott)': 0.1308, 'Recall (Bankrott)': 0.7297},
    {'Jahr': 3, 'Horizont': '3 Jahr(e)', 'Modell': 'Random Forest',      'ROC-AUC': 0.8836,
     'F1 (Bankrott)': 0.1928, 'Precision (Bankrott)': 0.8889, 'Recall (Bankrott)': 0.1081},
    {'Jahr': 3, 'Horizont': '3 Jahr(e)', 'Modell': 'XGBoost',            'ROC-AUC': 0.9018,
     'F1 (Bankrott)': 0.5414, 'Precision (Bankrott)': 0.6102, 'Recall (Bankrott)': 0.4865},
    {'Jahr': 4, 'Horizont': '2 Jahr(e)', 'Modell': 'Logistic Regression', 'ROC-AUC': 0.7308,
     'F1 (Bankrott)': 0.2424, 'Precision (Bankrott)': 0.1505, 'Recall (Bankrott)': 0.6234},
    {'Jahr': 4, 'Horizont': '2 Jahr(e)', 'Modell': 'Random Forest',      'ROC-AUC': 0.8625,
     'F1 (Bankrott)': 0.2826, 'Precision (Bankrott)': 0.8667, 'Recall (Bankrott)': 0.1688},
    {'Jahr': 4, 'Horizont': '2 Jahr(e)', 'Modell': 'XGBoost',            'ROC-AUC': 0.8981,
     'F1 (Bankrott)': 0.6861, 'Precision (Bankrott)': 0.7833, 'Recall (Bankrott)': 0.6104},
    {'Jahr': 5, 'Horizont': '1 Jahr(e)', 'Modell': 'Logistic Regression', 'ROC-AUC': 0.8510,
     'F1 (Bankrott)': 0.3852, 'Precision (Bankrott)': 0.2582, 'Recall (Bankrott)': 0.7581},
    {'Jahr': 5, 'Horizont': '1 Jahr(e)', 'Modell': 'Random Forest',      'ROC-AUC': 0.9468,
     'F1 (Bankrott)': 0.3133, 'Precision (Bankrott)': 0.6190, 'Recall (Bankrott)': 0.2097},
    {'Jahr': 5, 'Horizont': '1 Jahr(e)', 'Modell': 'XGBoost',            'ROC-AUC': 0.9593,
     'F1 (Bankrott)': 0.7049, 'Precision (Bankrott)': 0.7167, 'Recall (Bankrott)': 0.6935},
])

horizon_order = ['5 Jahr(e)', '4 Jahr(e)', '3 Jahr(e)',
                 '2 Jahr(e)', '1 Jahr(e)']

st.markdown("---")
st.subheader("Ergebnisse auf dem Testset")

tab1, tab2, tab3, tab4 = st.tabs(
    ["ROC-AUC", "F1 (Bankrott)", "Precision", "Recall"])

for tab, metric in zip([tab1, tab2, tab3, tab4],
                       ['ROC-AUC', 'F1 (Bankrott)', 'Precision (Bankrott)', 'Recall (Bankrott)']):
    with tab:
        fig = px.bar(all_results, x='Horizont', y=metric, color='Modell',
                     barmode='group', text=metric,
                     category_orders={'Horizont': horizon_order},
                     title=f'{metric} je Modell und Prognosehorizont (Testset)')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Vollstaendige Ergebnistabelle")
st.dataframe(all_results, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Zusammenfassung")

xgb = all_results[all_results['Modell'] == 'XGBoost']
col1, col2, col3, col4 = st.columns(4)
col1.metric("XGBoost Avg. ROC-AUC", f"{xgb['ROC-AUC'].mean():.3f}")
col2.metric("XGBoost Avg. F1", f"{xgb['F1 (Bankrott)'].mean():.3f}")
col3.metric("XGBoost Avg. Precision",
            f"{xgb['Precision (Bankrott)'].mean():.3f}")
col4.metric("XGBoost Avg. Recall", f"{xgb['Recall (Bankrott)'].mean():.3f}")

st.markdown("""
**Interpretation**

XGBoost übertrifft Logistische Regression und Random Forest konsistent über alle Prognosehorizonte.
Random Forest erzielt nahezu perfekte Precision, verpasst aber den Großteil der tatsächlichen
Insolvenzen (niedriger Recall). Die Logistische Regression findet viele Insolvenzen (hoher Recall),
produziert aber viele Fehlalarme (niedrige Precision).

Je kürzer der Prognosehorizont, desto besser die Vorhersagegüte. 1 Jahr vor Insolvenz
sind die finanziellen Warnsignale in den Kennzahlen bereits deutlich sichtbar.
""")
