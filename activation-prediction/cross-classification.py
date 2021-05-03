#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script finds the predictive performance on tcr x when training  only on tcr y

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

from preprocessing import (full_aa_features, get_aa_factors, get_aa_features,
                           get_dataset)


#%% training and evaluation
def train():
    df = get_dataset(normalization='AS')
    df['is_activated'] = df['activation'] > 46.9

    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    ) & (
        df['tcr'].isin(df.query('is_activated')['tcr'].unique())
    )]

    aa_features = get_aa_features()
    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)
    print('total features', fit_data.shape[1])

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    perf = []
    combs = itertools.product(tdf['tcr'].unique(), tdf['tcr'].unique())
    for train_tcr, test_tcr in tqdm(list(combs), ncols=50):
        train_mask = tdf['tcr'] == train_tcr
        test_mask = tdf['tcr'] == test_tcr

        # train and predict
        reg = RandomForestClassifier(
            n_estimators=1000,
            n_jobs=n_jobs,
        ).fit(fit_data[train_mask], tdf.loc[train_mask, 'is_activated'])
        test_preds = reg.predict_proba(fit_data[test_mask])

        # save performance
        pdf = tdf[test_mask][[
            'mut_pos', 'mut_ami', 'wild_activation',
            'activation', 'is_activated'
        ]]
        pdf['train_tcr'] = train_tcr
        pdf['test_tcr'] = test_tcr
        pdf['pred'] = test_preds[:, 1]
        perf.append(pdf)

    ppdf = pd.concat(perf)

    return ppdf


fname = 'results/cross-performance.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

#%% tcr distances

ddf = pd.read_csv(
    '../data/distances_activated_tcrs.csv'
).set_index('Unnamed: 0')

# copy OT1 distances to new names
ddf['OTI_PH'] = ddf['OT1']
ddf['LR_OTI_1'] = ddf['OT1']
ddf['LR_OTI_2'] = ddf['OT1']
ddf.loc['OTI_PH'] = ddf.loc['OT1']
ddf.loc['LR_OTI_1'] = ddf.loc['OT1']
ddf.loc['LR_OTI_2'] = ddf.loc['OT1']

ddf = ddf.reset_index().melt(
    'Unnamed: 0'
).rename(columns={
    'Unnamed: 0': 'train_tcr',
    'variable': 'test_tcr',
    'value': 'tcrdist',
}).applymap(
    lambda s: s.upper() if isinstance(s, str) else s
).replace({
    'ED161': 'ED16-1',
    'ED1630': 'ED16-30'
})

#%% correlation of activations

adf = pdf[
    pdf['train_tcr'] == pdf['test_tcr']
].sort_values(['mut_pos', 'mut_ami'])

cdf = pd.DataFrame([{
    'train_tcr': t1,
    'test_tcr': t2,
    'spearman': stats.spearmanr(
        adf[adf['test_tcr'] == t1]['activation'].values,
        adf[adf['test_tcr'] == t2]['activation'].values
    )[0]
} for t1, t2 in itertools.product(
    pdf['train_tcr'].unique(), pdf['train_tcr'].unique()
)])

#%% merge pair data

pairs = pdf.groupby(['train_tcr', 'test_tcr']).apply(lambda q: pd.Series({
    'auc': roc_auc_score(q['is_activated'], q['pred']),
    'aps': average_precision_score(q['is_activated'], q['pred']),
})).reset_index().merge(
    ddf, on=['train_tcr', 'test_tcr'],
).merge(cdf, on=['train_tcr', 'test_tcr'])

#%% pairgrid
pairs['TCRs'] = np.where(
    ~pairs['train_tcr'].isin(['B13', 'G6']) & ~pairs['test_tcr'].isin(['B13', 'G6']),
    'Others', 'B13 or G6',
)

g = sns.PairGrid(
    data=pairs.query('train_tcr!=test_tcr'),
    vars=['auc', 'tcrdist', 'spearman'],
    hue='TCRs', hue_order=['Others', 'B13 or G6'],
    height=2
)
g.map_upper(sns.regplot, marker='+', scatter_kws={'alpha': 0.4})
g.map_diag(sns.histplot, kde=True, edgecolor=None)
g.map_lower(sns.kdeplot, levels=4, color='.2')
g.map_lower(sns.scatterplot, alpha=0.4)
g.add_legend()
g.savefig('figures/cross-performance.pdf', dpi=192)

#%%

dd = pairs.query('test_tcr=="OTI_PH"')[[
    'train_tcr', 'auc', 'tcrdist'
]].rename(columns={
    'train_tcr': 'Other TCR',
    'auc': 'AUC (Test on OTI_PH)'
}).merge(
    pairs.query('train_tcr=="OTI_PH"')[[
        'test_tcr', 'auc',
    ]].rename(columns={
        'test_tcr': 'Other TCR',
        'auc': 'AUC (Train on OTI_PH)'
    }),
    on='Other TCR'
).sort_values('AUC (Test on OTI_PH)')

dd['TCR-closeness'] = 1 - dd['tcrdist']
g = sns.catplot(
    data=dd.drop(columns=['tcrdist']).melt('Other TCR', var_name='Metric'),
    x='Other TCR', y='value', kind='bar', hue='Metric', palette='pastel',
    aspect=2.5
)
#g.ax.tick_params(axis='x', rotation=90)
xl = g.ax.get_xlim()
g.ax.plot(xl, [0.96, 0.96], 'r--')
g.ax.set_xlim(xl)
g.ax.text(xl[0] + 0.1, 0.98, 'Leave-OT1-out AUC', c='r')
g.savefig('figures/cross-performance-ot1.pdf', dpi=192)
