import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

st.set_page_config(page_title="Vorhersage", layout="wide")
st.title("Interaktive Insolvenzvorhersage")

st.markdown("""
Geben Sie die Finanzkennzahlen eines Unternehmens ein und erhalten Sie eine Vorhersage
des Insolvenzrisikos. Felder die nicht ausgefuellt werden, erhalten automatisch den
Medianwert aus dem Trainingsset des jeweiligen Jahres.
""")

MODELS_DIR = 'models'

if not os.path.exists(os.path.join(MODELS_DIR, 'model_year1.pkl')):
    st.warning("Modelle noch nicht trainiert. Bitte zuerst 'python train_models.py' ausfuehren.")
    st.stop()

@st.cache_resource
def load_artifact(year):
    with open(os.path.join(MODELS_DIR, f'model_year{year}.pkl'), 'rb') as f:
        return pickle.load(f)

st.markdown("---")

col_year, col_info = st.columns([1, 2])
with col_year:
    year = st.selectbox(
        "Prognosejahr auswählen",
        options=[1, 2, 3, 4, 5],
        format_func=lambda y: f"Jahr {y}  ({6-y} Jahr(e) bis Insolvenz)",
        index=4,
    )
with col_info:
    horizon_info = {
        1: "Die Finanzdaten stammen aus Jahr 1. Das Modell sagt vorher, ob das Unternehmen in 5 Jahren insolvent wird.",
        2: "Die Finanzdaten stammen aus Jahr 2. Das Modell sagt vorher, ob das Unternehmen in 4 Jahren insolvent wird.",
        3: "Die Finanzdaten stammen aus Jahr 3. Das Modell sagt vorher, ob das Unternehmen in 3 Jahren insolvent wird.",
        4: "Die Finanzdaten stammen aus Jahr 4. Das Modell sagt vorher, ob das Unternehmen in 2 Jahren insolvent wird.",
        5: "Die Finanzdaten stammen aus Jahr 5. Das Modell sagt vorher, ob das Unternehmen im naechsten Jahr insolvent wird.",
    }
    perf = {1: (0.974, 0.785), 2: (0.892, 0.635), 3: (0.902, 0.541),
            4: (0.898, 0.686), 5: (0.959, 0.705)}
    st.info(horizon_info[year])
    c1, c2 = st.columns(2)
    c1.metric("Modell ROC-AUC", f"{perf[year][0]:.3f}")
    c2.metric("Modell F1 (Bankrott)", f"{perf[year][1]:.3f}")

artifact = load_artifact(year)

# Wichtigste Merkmale mit deutschen Bezeichnungen
KEY_FEATURES = [
    ('X1_net_profit_total_assets',               'X1: Nettoprofit / Gesamtvermoegen',               -0.5,  0.5),
    ('X2_total_liabilities_total_assets',         'X2: Gesamtschulden / Gesamtvermoegen',             0.0,  2.0),
    ('X3_working_capital_total_assets',           'X3: Working Capital / Gesamtvermoegen',           -1.0,  1.0),
    ('X4_current_assets_short_term_liabilities',  'X4: Umlaufvermoegen / Kurzfristige Schulden',      0.0,  5.0),
    ('X7_EBIT_total_assets',                      'X7: EBIT / Gesamtvermoegen',                      -0.5,  0.5),
    ('X10_equity_total_assets',                   'X10: Eigenkapital / Gesamtvermoegen',             -0.5,  1.0),
    ('X18_gross_profit_total_assets',             'X18: Bruttogewinn / Gesamtvermoegen',             -0.5,  0.5),
    ('X22_operating_profit_total_assets',         'X22: Betriebsgewinn / Gesamtvermoegen',           -0.5,  0.5),
    ('X25_equity_share_capital_total_assets',     'X25: (EK - Stammkapital) / Gesamtvermoegen',      -0.5,  1.0),
    ('X29_log_total_assets',                      'X29: Logarithmus Gesamtvermoegen',                 0.0, 10.0),
    ('X38_constant_capital_total_assets',         'X38: Langfristiges Kapital / Gesamtvermoegen',     0.0,  1.5),
    ('X51_short_term_liabilities_total_assets',   'X51: Kurzfristige Schulden / Gesamtvermoegen',     0.0,  1.0),
]

st.markdown("---")
st.subheader("Finanzkennzahlen eingeben")
st.markdown("Alle Felder sind mit dem Medianwert des Trainingssets vorbelegt. Passen Sie die Werte an das zu pruefende Unternehmen an.")

with st.form("prediction_form"):
    user_inputs = {}
    cols = st.columns(3)
    for i, (feat, label, min_v, max_v) in enumerate(KEY_FEATURES):
        if feat in artifact['feature_cols']:
            default = float(artifact['train_medians'].get(feat, 0.0))
            user_inputs[feat] = cols[i % 3].number_input(
                label, value=default, format="%.4f",
                help=f"Traingsmedian: {default:.4f}")

    st.markdown("Alle weiteren Merkmale (nicht angezeigt) werden automatisch mit dem Trainingsmedian belegt.")
    submitted = st.form_submit_button("Vorhersage starten", type="primary")

if submitted:
    # Vollstaendigen Feature-Vektor aus Trainingsmedians erstellen
    feature_vector = {col: artifact['train_medians'].get(col, 0.0)
                      for col in artifact['feature_cols']}
    # Benutzereingaben einsetzen
    feature_vector.update(user_inputs)

    X_input = pd.DataFrame([feature_vector])[artifact['feature_cols']]
    X_scaled = artifact['scaler'].transform(X_input)

    proba = artifact['model'].predict_proba(X_scaled)[0][1]
    pred  = int(artifact['model'].predict(X_scaled)[0])

    st.markdown("---")
    st.subheader("Ergebnis")

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if pred == 1:
            st.error(f"Vorhersage: BANKROTT")
            st.markdown(f"**Bankrott-Wahrscheinlichkeit: {proba:.1%}**")
            st.markdown(f"Das Modell stuft dieses Unternehmen mit einer Wahrscheinlichkeit "
                        f"von {proba:.1%} als insolvenzgefaehrdet ein.")
        else:
            st.success(f"Vorhersage: NICHT BANKROTT")
            st.markdown(f"**Bankrott-Wahrscheinlichkeit: {proba:.1%}**")
            st.markdown(f"Das Modell stuft dieses Unternehmen als nicht insolvenzgefaehrdet ein "
                        f"(Wahrscheinlichkeit: {proba:.1%}).")

        st.markdown(f"""
**Hinweis zur Interpretation:**

Der Schwellenwert fuer die Klassifikation liegt bei 0.5.
Eine Bankrott-Wahrscheinlichkeit ueber 50 Prozent fuehrt zur Klassifikation als bankrott.
In der Praxis kann dieser Schwellenwert je nach Risikopraeferenz angepasst werden.
""")

    with col_gauge:
        gauge_df = pd.DataFrame({'Kategorie': ['Bankrott-Risiko'], 'Wert': [proba * 100]})
        fig = px.bar(gauge_df, x='Kategorie', y='Wert',
                     color='Wert', color_continuous_scale='RdYlGn_r',
                     range_color=[0, 100], range_y=[0, 100],
                     text='Wert',
                     title=f'Bankrott-Wahrscheinlichkeit: {proba:.1%}')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=350, xaxis_title='', yaxis_title='Wahrscheinlichkeit (%)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Eingabeuebersicht")
    input_display = {label: f"{user_inputs.get(feat, artifact['train_medians'].get(feat, 0)):.4f}"
                     for feat, label, _, _ in KEY_FEATURES if feat in artifact['feature_cols']}
    st.dataframe(pd.DataFrame(input_display.items(), columns=['Merkmal', 'Wert']),
                 use_container_width=True, hide_index=True)
