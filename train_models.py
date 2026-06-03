"""
Trainiert XGBoost-Modelle fuer alle 5 Prognosejahre und speichert
Modell, Imputer, Scaler und Trainingsmediane als Pickle-Dateien.

Ausfuehren: python train_models.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_DIR     = 'data'
MODELS_DIR   = 'models'
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

feature_names = {
    'Attr1':  'X1_net_profit_total_assets',
    'Attr2':  'X2_total_liabilities_total_assets',
    'Attr3':  'X3_working_capital_total_assets',
    'Attr4':  'X4_current_assets_short_term_liabilities',
    'Attr5':  'X5_cash_flow_days',
    'Attr6':  'X6_retained_earnings_total_assets',
    'Attr7':  'X7_EBIT_total_assets',
    'Attr8':  'X8_book_value_equity_total_liabilities',
    'Attr9':  'X9_sales_total_assets',
    'Attr10': 'X10_equity_total_assets',
    'Attr11': 'X11_gross_profit_extraord_finexp_total_assets',
    'Attr12': 'X12_gross_profit_short_term_liabilities',
    'Attr13': 'X13_gross_profit_depr_sales',
    'Attr14': 'X14_gross_profit_interest_total_assets',
    'Attr15': 'X15_total_liabilities_days_gross_profit_depr',
    'Attr16': 'X16_gross_profit_depr_total_liabilities',
    'Attr17': 'X17_total_assets_total_liabilities',
    'Attr18': 'X18_gross_profit_total_assets',
    'Attr19': 'X19_gross_profit_sales',
    'Attr20': 'X20_inventory_days_sales',
    'Attr21': 'X21_sales_growth',
    'Attr22': 'X22_operating_profit_total_assets',
    'Attr23': 'X23_net_profit_sales',
    'Attr24': 'X24_gross_profit_3y_total_assets',
    'Attr25': 'X25_equity_share_capital_total_assets',
    'Attr26': 'X26_net_profit_depr_total_liabilities',
    'Attr27': 'X27_operating_profit_financial_expenses',
    'Attr28': 'X28_working_capital_fixed_assets',
    'Attr29': 'X29_log_total_assets',
    'Attr30': 'X30_total_liabilities_cash_sales',
    'Attr31': 'X31_gross_profit_interest_sales',
    'Attr32': 'X32_current_liabilities_days_cost_products',
    'Attr33': 'X33_operating_expenses_short_term_liabilities',
    'Attr34': 'X34_operating_expenses_total_liabilities',
    'Attr35': 'X35_profit_on_sales_total_assets',
    'Attr36': 'X36_total_sales_total_assets',
    'Attr37': 'X37_current_assets_inv_long_term_liabilities',
    'Attr38': 'X38_constant_capital_total_assets',
    'Attr39': 'X39_profit_on_sales_sales',
    'Attr40': 'X40_current_assets_inv_rec_short_term_liab',
    'Attr41': 'X41_total_liabilities_operating_days',
    'Attr42': 'X42_operating_profit_sales',
    'Attr43': 'X43_receivables_inventory_turnover_days',
    'Attr44': 'X44_receivables_days_sales',
    'Attr45': 'X45_net_profit_inventory',
    'Attr46': 'X46_current_assets_inv_short_term_liabilities',
    'Attr47': 'X47_inventory_days_cost_products',
    'Attr48': 'X48_EBITDA_total_assets',
    'Attr49': 'X49_EBITDA_sales',
    'Attr50': 'X50_current_assets_total_liabilities',
    'Attr51': 'X51_short_term_liabilities_total_assets',
    'Attr52': 'X52_short_term_liabilities_days_cost_products',
    'Attr53': 'X53_equity_fixed_assets',
    'Attr54': 'X54_constant_capital_fixed_assets',
    'Attr55': 'X55_working_capital',
    'Attr56': 'X56_sales_gross_margin',
    'Attr57': 'X57_liquidity_ratio',
    'Attr58': 'X58_total_costs_total_sales',
    'Attr59': 'X59_long_term_liabilities_equity',
    'Attr60': 'X60_sales_inventory',
    'Attr61': 'X61_sales_receivables',
    'Attr62': 'X62_short_term_liabilities_days_sales',
    'Attr63': 'X63_sales_short_term_liabilities',
    'Attr64': 'X64_sales_fixed_assets',
}

DROP_FEATURE = 'X37_current_assets_inv_long_term_liabilities'
MISSING_IND  = 'X21_sales_growth'

print('Lade Daten ...')
dfs = []
for year in range(1, 6):
    data, _ = arff.loadarff(f'{DATA_DIR}/{year}year.arff')
    df = pd.DataFrame(data)
    df['year'] = year
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all['class'] = df_all['class'].apply(
    lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
df_all.rename(columns=feature_names, inplace=True)

results = []

for year in range(1, 6):
    print(f'Trainiere Modell fuer Jahr {year} ...')
    df_year = df_all[df_all['year'] == year].copy()
    raw_feat_cols = [c for c in df_year.columns if c not in ['class', 'year']]
    X = df_year[raw_feat_cols]
    y = df_year['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)

    for Xs in [X_train, X_test]:
        Xs.drop(columns=[DROP_FEATURE], inplace=True, errors='ignore')
    for Xs in [X_train, X_test]:
        Xs[f'{MISSING_IND}_missing'] = Xs[MISSING_IND].isnull().astype(int)

    num_cols = [c for c in X_train.columns if c != f'{MISSING_IND}_missing']

    imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols]  = imputer.transform(X_test[num_cols])

    train_medians = X_train.median().to_dict()
    final_cols    = list(X_train.columns)

    scaler    = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pos_weight     = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = XGBClassifier(
        scale_pos_weight=pos_weight, n_estimators=200,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    results.append({
        'Jahr':                 year,
        'Horizont':             f'{6 - year} Jahr(e)',
        'ROC-AUC':              round(roc_auc_score(y_test, y_proba), 4),
        'F1 (Bankrott)':        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        'Precision (Bankrott)': round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        'Recall (Bankrott)':    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
    })

    artifact = {
        'model':         model,
        'imputer':       imputer,
        'scaler':        scaler,
        'feature_cols':  final_cols,
        'num_cols':      num_cols,
        'train_medians': train_medians,
        'year':          year,
    }

    path = os.path.join(MODELS_DIR, f'model_year{year}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f'  Jahr {year}: AUC={results[-1]["ROC-AUC"]}, F1={results[-1]["F1 (Bankrott)"]}')

results_df = pd.DataFrame(results)
with open(os.path.join(MODELS_DIR, 'results.pkl'), 'wb') as f:
    pickle.dump(results_df, f)

print('\nAlle Modelle und Ergebnisse gespeichert.')
print(results_df.to_string(index=False))
