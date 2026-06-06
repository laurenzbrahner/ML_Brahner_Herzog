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
des Insolvenzrisikos. Alle Felder sind mit dem Trainingsmedian vorbelegt.
Das Modell wurde mit 63 Finanzkennzahlen und einem Missing-Indikator für X21 trainiert (64 Merkmale gesamt).
""")

MODELS_DIR = 'models'

if not os.path.exists(os.path.join(MODELS_DIR, 'model_year1.pkl')):
    st.warning("Modelle noch nicht trainiert. Bitte zuerst 'python train_models.py' ausführen.")
    st.stop()

@st.cache_resource
def load_artifact(year):
    with open(os.path.join(MODELS_DIR, f'model_year{year}.pkl'), 'rb') as f:
        return pickle.load(f)

# Alle Features mit deutschen Kurzbezeichnungen und Kategorie
ALL_FEATURES = {
    'X1_net_profit_total_assets':                  ('X1:  Nettoprofit / Gesamtvermoegen',                      'Profitabilitaet'),
    'X7_EBIT_total_assets':                        ('X7:  EBIT / Gesamtvermoegen',                             'Profitabilitaet'),
    'X11_gross_profit_extraord_finexp_total_assets':('X11: Bruttogewinn+Sond.+Finanzaufw. / Gesamt.',          'Profitabilitaet'),
    'X13_gross_profit_depr_sales':                 ('X13: (Bruttogewinn+Abschr.) / Umsatz',                    'Profitabilitaet'),
    'X14_gross_profit_interest_total_assets':      ('X14: (Bruttogewinn+Zinsen) / Gesamtvermoegen',            'Profitabilitaet'),
    'X18_gross_profit_total_assets':               ('X18: Bruttogewinn / Gesamtvermoegen',                     'Profitabilitaet'),
    'X19_gross_profit_sales':                      ('X19: Bruttogewinn / Umsatz',                              'Profitabilitaet'),
    'X22_operating_profit_total_assets':           ('X22: Betriebsgewinn / Gesamtvermoegen',                   'Profitabilitaet'),
    'X23_net_profit_sales':                        ('X23: Nettoprofit / Umsatz',                               'Profitabilitaet'),
    'X24_gross_profit_3y_total_assets':            ('X24: Bruttogewinn (3J.) / Gesamtvermoegen',               'Profitabilitaet'),
    'X35_profit_on_sales_total_assets':            ('X35: Vertriebsgewinn / Gesamtvermoegen',                  'Profitabilitaet'),
    'X39_profit_on_sales_sales':                   ('X39: Vertriebsgewinn / Umsatz',                           'Profitabilitaet'),
    'X42_operating_profit_sales':                  ('X42: Betriebsgewinn / Umsatz',                            'Profitabilitaet'),
    'X48_EBITDA_total_assets':                     ('X48: EBITDA / Gesamtvermoegen',                           'Profitabilitaet'),
    'X49_EBITDA_sales':                            ('X49: EBITDA / Umsatz',                                    'Profitabilitaet'),
    'X56_sales_gross_margin':                      ('X56: Bruttomarge (Umsatz-Kosten) / Umsatz',               'Profitabilitaet'),

    'X2_total_liabilities_total_assets':           ('X2:  Gesamtschulden / Gesamtvermoegen',                   'Kapitalstruktur'),
    'X8_book_value_equity_total_liabilities':      ('X8:  Buchwert EK / Gesamtschulden',                       'Kapitalstruktur'),
    'X10_equity_total_assets':                     ('X10: Eigenkapital / Gesamtvermoegen',                     'Kapitalstruktur'),
    'X17_total_assets_total_liabilities':          ('X17: Gesamtvermoegen / Gesamtschulden',                   'Kapitalstruktur'),
    'X25_equity_share_capital_total_assets':       ('X25: (EK - Stammkapital) / Gesamtvermoegen',              'Kapitalstruktur'),
    'X26_net_profit_depr_total_liabilities':       ('X26: (Nettoprofit+Abschr.) / Gesamtschulden',             'Kapitalstruktur'),
    'X38_constant_capital_total_assets':           ('X38: Langfristiges Kapital / Gesamtvermoegen',            'Kapitalstruktur'),
    'X51_short_term_liabilities_total_assets':     ('X51: Kurzfristige Schulden / Gesamtvermoegen',            'Kapitalstruktur'),
    'X53_equity_fixed_assets':                     ('X53: Eigenkapital / Anlagevermoegen',                     'Kapitalstruktur'),
    'X54_constant_capital_fixed_assets':           ('X54: Langfristiges Kapital / Anlagevermoegen',            'Kapitalstruktur'),
    'X59_long_term_liabilities_equity':            ('X59: Langfristige Schulden / Eigenkapital',               'Kapitalstruktur'),

    'X3_working_capital_total_assets':             ('X3:  Working Capital / Gesamtvermoegen',                  'Liquiditaet'),
    'X4_current_assets_short_term_liabilities':    ('X4:  Umlaufvermoegen / Kurzfristige Schulden',            'Liquiditaet'),
    'X5_cash_flow_days':                           ('X5:  Cash-Flow in Tagen',                                 'Liquiditaet'),
    'X6_retained_earnings_total_assets':           ('X6:  Gewinnruecklagen / Gesamtvermoegen',                 'Liquiditaet'),
    'X40_current_assets_inv_rec_short_term_liab':  ('X40: (Umlauf.-Vorr.-Ford.) / Kurzfr. Schulden',          'Liquiditaet'),
    'X46_current_assets_inv_short_term_liabilities':('X46: (Umlaufv.-Vorraete) / Kurzfr. Schulden',           'Liquiditaet'),
    'X50_current_assets_total_liabilities':        ('X50: Umlaufvermoegen / Gesamtschulden',                   'Liquiditaet'),
    'X55_working_capital':                         ('X55: Working Capital (absolut)',                           'Liquiditaet'),
    'X57_liquidity_ratio':                         ('X57: Liquiditaetsquote',                                  'Liquiditaet'),

    'X9_sales_total_assets':                       ('X9:  Umsatz / Gesamtvermoegen',                           'Aktivitaet'),
    'X20_inventory_days_sales':                    ('X20: (Vorraete x 365) / Umsatz',                          'Aktivitaet'),
    'X21_sales_growth':                            ('X21: Umsatzwachstum (n/n-1)',                              'Aktivitaet'),
    'X27_operating_profit_financial_expenses':     ('X27: Betriebsgewinn / Finanzaufwendungen',                'Aktivitaet'),
    'X28_working_capital_fixed_assets':            ('X28: Working Capital / Anlagevermoegen',                  'Aktivitaet'),
    'X29_log_total_assets':                        ('X29: Logarithmus Gesamtvermoegen',                        'Aktivitaet'),
    'X36_total_sales_total_assets':                ('X36: Gesamtumsatz / Gesamtvermoegen',                     'Aktivitaet'),
    'X43_receivables_inventory_turnover_days':     ('X43: Umschlagsdauer Ford.+Vorraete (Tage)',               'Aktivitaet'),
    'X44_receivables_days_sales':                  ('X44: (Forderungen x 365) / Umsatz',                       'Aktivitaet'),
    'X45_net_profit_inventory':                    ('X45: Nettoprofit / Vorraete',                             'Aktivitaet'),
    'X47_inventory_days_cost_products':            ('X47: (Vorraete x 365) / Herstellkosten',                  'Aktivitaet'),
    'X58_total_costs_total_sales':                 ('X58: Gesamtkosten / Gesamtumsatz',                        'Aktivitaet'),
    'X60_sales_inventory':                         ('X60: Umsatz / Vorraete',                                  'Aktivitaet'),
    'X61_sales_receivables':                       ('X61: Umsatz / Forderungen',                               'Aktivitaet'),
    'X64_sales_fixed_assets':                      ('X64: Umsatz / Anlagevermoegen',                           'Aktivitaet'),

    'X12_gross_profit_short_term_liabilities':     ('X12: Bruttogewinn / Kurzfristige Schulden',               'Kurzfr. Verbindlichkeiten'),
    'X15_total_liabilities_days_gross_profit_depr':('X15: (Schulden x 365) / (BG+Abschr.)',                   'Kurzfr. Verbindlichkeiten'),
    'X16_gross_profit_depr_total_liabilities':     ('X16: (Bruttogewinn+Abschr.) / Gesamtschulden',            'Kurzfr. Verbindlichkeiten'),
    'X30_total_liabilities_cash_sales':            ('X30: (Schulden-Cash) / Umsatz',                           'Kurzfr. Verbindlichkeiten'),
    'X31_gross_profit_interest_sales':             ('X31: (Bruttogewinn+Zinsen) / Umsatz',                     'Kurzfr. Verbindlichkeiten'),
    'X32_current_liabilities_days_cost_products':  ('X32: (Kurzfr. Schulden x 365) / Herstellkosten',         'Kurzfr. Verbindlichkeiten'),
    'X33_operating_expenses_short_term_liabilities':('X33: Betriebskosten / Kurzfristige Schulden',            'Kurzfr. Verbindlichkeiten'),
    'X34_operating_expenses_total_liabilities':    ('X34: Betriebskosten / Gesamtschulden',                    'Kurzfr. Verbindlichkeiten'),
    'X37_current_assets_inv_long_term_liabilities':('X37: (Umlaufv.-Vorr.) / Langfristige Schulden',           'Kurzfr. Verbindlichkeiten'),
    'X41_total_liabilities_operating_days':        ('X41: Schulden / (Betriebsgewinn+Abschr.) x 12/365',      'Kurzfr. Verbindlichkeiten'),
    'X52_short_term_liabilities_days_cost_products':('X52: (Kurzfr. Schulden x 365) / Herstellkosten',        'Kurzfr. Verbindlichkeiten'),
    'X62_short_term_liabilities_days_sales':       ('X62: (Kurzfr. Schulden x 365) / Umsatz',                 'Kurzfr. Verbindlichkeiten'),
    'X63_sales_short_term_liabilities':            ('X63: Umsatz / Kurzfristige Schulden',                     'Kurzfr. Verbindlichkeiten'),
}

CATEGORIES = ['Profitabilitaet', 'Kapitalstruktur', 'Liquiditaet',
              'Aktivitaet', 'Kurzfr. Verbindlichkeiten']

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
        1: "Vorhersage, ob das Unternehmen in 5 Jahren insolvent wird.",
        2: "Vorhersage, ob das Unternehmen in 4 Jahren insolvent wird.",
        3: "Vorhersage, ob das Unternehmen in 3 Jahren insolvent wird.",
        4: "Vorhersage, ob das Unternehmen in 2 Jahren insolvent wird.",
        5: "Vorhersage, ob das Unternehmen im nächsten Jahr insolvent wird.",
    }
    perf = {1: (0.974, 0.785), 2: (0.892, 0.635), 3: (0.902, 0.541),
            4: (0.898, 0.686), 5: (0.959, 0.705)}
    st.info(horizon_info[year])
    c1, c2 = st.columns(2)
    c1.metric("Modell ROC-AUC", f"{perf[year][0]:.3f}")
    c2.metric("Modell F1 (Bankrott)", f"{perf[year][1]:.3f}")

artifact = load_artifact(year)

st.markdown("---")
st.subheader("Finanzkennzahlen eingeben")
st.markdown("Alle Felder sind mit dem Trainingsmedian vorbelegt. Passen Sie die Werte an.")

user_inputs = {}

with st.form("prediction_form"):
    for cat in CATEGORIES:
        cat_features = {k: v for k, v in ALL_FEATURES.items()
                        if v[1] == cat and k in artifact['feature_cols']}
        if not cat_features:
            continue

        with st.expander(f"{cat} ({len(cat_features)} Merkmale)", expanded=(cat == 'Profitabilitaet')):
            cols = st.columns(3)
            for i, (feat, (label, _)) in enumerate(cat_features.items()):
                default = float(artifact['train_medians'].get(feat, 0.0))
                user_inputs[feat] = cols[i % 3].number_input(
                    label, value=default, format="%.4f",
                    help=f"Trainingsmedian: {default:.4f}")

    # X21_missing Indikator
    x21_key = 'X21_sales_growth_missing'
    if x21_key in artifact['feature_cols']:
        user_inputs[x21_key] = 0

    submitted = st.form_submit_button("Vorhersage starten", type="primary")

if submitted:
    feature_vector = {col: artifact['train_medians'].get(col, 0.0)
                      for col in artifact['feature_cols']}
    feature_vector.update(user_inputs)

    X_input  = pd.DataFrame([feature_vector])[artifact['feature_cols']]
    X_scaled = artifact['scaler'].transform(X_input)

    proba = artifact['model'].predict_proba(X_scaled)[0][1]
    pred  = int(artifact['model'].predict(X_scaled)[0])

    st.markdown("---")
    st.subheader("Ergebnis")

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if pred == 1:
            st.error("Vorhersage: BANKROTT")
            st.markdown(f"**Bankrott-Wahrscheinlichkeit: {proba:.1%}**")
        else:
            st.success("Vorhersage: NICHT BANKROTT")
            st.markdown(f"**Bankrott-Wahrscheinlichkeit: {proba:.1%}**")

        st.markdown(f"Der Entscheidungsschwellenwert liegt bei 50 %. "
                    f"Das Modell basiert auf 63 Finanzkennzahlen + X21_missing-Indikator.")

    with col_gauge:
        fig = px.bar(x=['Bankrott-Wahrscheinlichkeit'], y=[proba * 100],
                     color=[proba * 100], color_continuous_scale='RdYlGn_r',
                     range_color=[0, 100], range_y=[0, 100],
                     text=[f"{proba:.1%}"],
                     title=f'Bankrott-Wahrscheinlichkeit: {proba:.1%}')
        fig.update_traces(textposition='outside')
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=350, xaxis_title='', yaxis_title='Wahrscheinlichkeit (%)')
        st.plotly_chart(fig, use_container_width=True)
