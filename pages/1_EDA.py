import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import arff
import plotly.express as px

st.set_page_config(page_title="EDA", layout="wide")
st.title("Explorative Datenanalyse")

FEATURE_NAMES = {
    'Attr1':'X1_net_profit_total_assets','Attr2':'X2_total_liabilities_total_assets',
    'Attr3':'X3_working_capital_total_assets','Attr4':'X4_current_assets_short_term_liabilities',
    'Attr5':'X5_cash_flow_days','Attr6':'X6_retained_earnings_total_assets',
    'Attr7':'X7_EBIT_total_assets','Attr8':'X8_book_value_equity_total_liabilities',
    'Attr9':'X9_sales_total_assets','Attr10':'X10_equity_total_assets',
    'Attr11':'X11_gross_profit_extraord_finexp_total_assets','Attr12':'X12_gross_profit_short_term_liabilities',
    'Attr13':'X13_gross_profit_depr_sales','Attr14':'X14_gross_profit_interest_total_assets',
    'Attr15':'X15_total_liabilities_days_gross_profit_depr','Attr16':'X16_gross_profit_depr_total_liabilities',
    'Attr17':'X17_total_assets_total_liabilities','Attr18':'X18_gross_profit_total_assets',
    'Attr19':'X19_gross_profit_sales','Attr20':'X20_inventory_days_sales',
    'Attr21':'X21_sales_growth','Attr22':'X22_operating_profit_total_assets',
    'Attr23':'X23_net_profit_sales','Attr24':'X24_gross_profit_3y_total_assets',
    'Attr25':'X25_equity_share_capital_total_assets','Attr26':'X26_net_profit_depr_total_liabilities',
    'Attr27':'X27_operating_profit_financial_expenses','Attr28':'X28_working_capital_fixed_assets',
    'Attr29':'X29_log_total_assets','Attr30':'X30_total_liabilities_cash_sales',
    'Attr31':'X31_gross_profit_interest_sales','Attr32':'X32_current_liabilities_days_cost_products',
    'Attr33':'X33_operating_expenses_short_term_liabilities','Attr34':'X34_operating_expenses_total_liabilities',
    'Attr35':'X35_profit_on_sales_total_assets','Attr36':'X36_total_sales_total_assets',
    'Attr37':'X37_current_assets_inv_long_term_liabilities','Attr38':'X38_constant_capital_total_assets',
    'Attr39':'X39_profit_on_sales_sales','Attr40':'X40_current_assets_inv_rec_short_term_liab',
    'Attr41':'X41_total_liabilities_operating_days','Attr42':'X42_operating_profit_sales',
    'Attr43':'X43_receivables_inventory_turnover_days','Attr44':'X44_receivables_days_sales',
    'Attr45':'X45_net_profit_inventory','Attr46':'X46_current_assets_inv_short_term_liabilities',
    'Attr47':'X47_inventory_days_cost_products','Attr48':'X48_EBITDA_total_assets',
    'Attr49':'X49_EBITDA_sales','Attr50':'X50_current_assets_total_liabilities',
    'Attr51':'X51_short_term_liabilities_total_assets','Attr52':'X52_short_term_liabilities_days_cost_products',
    'Attr53':'X53_equity_fixed_assets','Attr54':'X54_constant_capital_fixed_assets',
    'Attr55':'X55_working_capital','Attr56':'X56_sales_gross_margin',
    'Attr57':'X57_liquidity_ratio','Attr58':'X58_total_costs_total_sales',
    'Attr59':'X59_long_term_liabilities_equity','Attr60':'X60_sales_inventory',
    'Attr61':'X61_sales_receivables','Attr62':'X62_short_term_liabilities_days_sales',
    'Attr63':'X63_sales_short_term_liabilities','Attr64':'X64_sales_fixed_assets',
}

@st.cache_data
def load_data():
    dfs = []
    for year in range(1, 6):
        data, _ = arff.loadarff(f'data/{year}year.arff')
        df = pd.DataFrame(data)
        df['year'] = year
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['class'] = df_all['class'].apply(
        lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
    df_all.rename(columns=FEATURE_NAMES, inplace=True)
    return df_all

df_all = load_data()
feat_cols = [c for c in df_all.columns if c not in ['class', 'year']]

st.markdown("""
Der Datensatz umfasst Jahresabschlussdaten polnischer Unternehmen aus dem Zeitraum 2000 bis 2013.
Fünf Teildatensätze entsprechen fünf unterschiedlichen Prognosehorizonten.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Beobachtungen gesamt", f"{len(df_all):,}")
col2.metric("Merkmale", len(feat_cols))
col3.metric("Bankrott-Anteil", f"{df_all['class'].mean():.1%}")

st.markdown("---")

st.subheader("Klassenverteilung")
col1, col2 = st.columns(2)

with col1:
    cc = df_all['class'].value_counts().reset_index()
    cc.columns = ['class', 'count']
    cc['Klasse'] = cc['class'].map({0: 'Nicht bankrott', 1: 'Bankrott'})
    fig = px.pie(cc, names='Klasse', values='count',
                 color='Klasse',
                 color_discrete_map={'Nicht bankrott': '#2196F3', 'Bankrott': '#F44336'},
                 hole=0.4, title='Gesamtverteilung')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ys = df_all.groupby('year')['class'].agg(['sum', 'count']).reset_index()
    ys['Bankrott-Rate (%)'] = (ys['sum'] / ys['count'] * 100).round(2)
    ys['Horizont'] = ys['year'].map(
        {1: '5 Jahre', 2: '4 Jahre', 3: '3 Jahre', 4: '2 Jahre', 5: '1 Jahr'})
    fig = px.bar(ys, x='Horizont', y='Bankrott-Rate (%)',
                 text='Bankrott-Rate (%)', color='Bankrott-Rate (%)',
                 color_continuous_scale='Reds',
                 title='Bankrott-Rate je Prognosehorizont')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Fehlende Werte")

missing = (df_all[feat_cols].isnull().sum() / len(df_all) * 100).sort_values(ascending=True)
missing = missing[missing > 0].reset_index()
missing.columns = ['Feature', 'Fehlend (%)']
missing['Feature_kurz'] = missing['Feature'].str.split('_').str[0]

fig = px.bar(missing, x='Fehlend (%)', y='Feature_kurz', orientation='h',
             color='Fehlend (%)', color_continuous_scale='Reds',
             text='Fehlend (%)', title='Anteil fehlender Werte pro Merkmal')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(height=650, coloraxis_showscale=False, yaxis_title='Merkmal')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Top-20 Merkmale nach Trennkraft (Median-Differenz)")
st.markdown("Positive Werte bedeuten, dass bankrotte Unternehmen höhere Werte aufweisen. Negative Werte bedeuten, dass gesunde Unternehmen höhere Werte aufweisen.")

medians = df_all.groupby('class')[feat_cols].median().T
medians.columns = ['Nicht bankrott', 'Bankrott']
medians['diff_norm'] = (medians['Bankrott'] - medians['Nicht bankrott']) / df_all[feat_cols].std()
top20 = medians.reindex(medians['diff_norm'].abs().sort_values(ascending=True).index).tail(20).reset_index()
top20.columns = ['Feature', 'Nicht bankrott', 'Bankrott', 'Norm. Differenz']
top20['Feature_kurz'] = top20['Feature'].str.split('_').str[0]

top20['Farbe'] = top20['Norm. Differenz'].apply(
    lambda x: 'Bankrott hoeher' if x > 0 else 'Gesund hoeher')

fig = px.bar(top20, x='Norm. Differenz', y='Feature_kurz', orientation='h',
             color='Farbe',
             color_discrete_map={'Bankrott hoeher': '#E53935', 'Gesund hoeher': '#1E88E5'},
             title='Normierte Median-Differenz (Bankrott vs. Nicht bankrott)')
fig.update_layout(height=500, yaxis_title='Merkmal',
                  legend=dict(title='', orientation='h', y=1.05))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Ausreißer-Analyse")
st.markdown("Anteil extremer Werte pro Merkmal (mehr als 3-facher IQR-Abstand).")

q1 = df_all[feat_cols].quantile(0.25)
q3 = df_all[feat_cols].quantile(0.75)
iqr = q3 - q1
outlier_pct = (((df_all[feat_cols] < (q1 - 3*iqr)) | (df_all[feat_cols] > (q3 + 3*iqr))).sum()
               / len(df_all) * 100).sort_values(ascending=False).head(20).reset_index()
outlier_pct.columns = ['Feature', 'Ausreißer (%)']
outlier_pct['Feature_kurz'] = outlier_pct['Feature'].str.split('_').str[0]

fig = px.bar(outlier_pct, x='Feature_kurz', y='Ausreißer (%)',
             color='Ausreißer (%)', color_continuous_scale='Oranges',
             text='Ausreißer (%)', title='Top-20 Merkmale mit extremen Ausreißern')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(height=420, coloraxis_showscale=False, xaxis_title='Merkmal')
st.plotly_chart(fig, use_container_width=True)
