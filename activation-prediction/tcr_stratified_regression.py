#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this trains a random forest validating on a held-out receptor


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from preprocessing import full_aa_features, get_aa_features, get_dataset, get_tumor_dataset

# %% training and evaluation

def train():
    if epitope == 'VPSVWRSSL':
        df = get_tumor_dataset()
    else:
        df = get_dataset(normalization='AS')
    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    ) & (
        df['tcr'].isin(df.query('activation > 15')['tcr'].unique())
    )]

    aa_features = get_aa_features()

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    fit_data = full_aa_features(tdf, aa_features, include_tcr=True, base_peptide=epitope)
    print('total features', fit_data.shape[1])

    perf = []
    for norm in tqdm(tdf['normalization'].unique(), ncols=50):
        data_mask = tdf['normalization'] == norm
        for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
            train_mask = data_mask & (tdf['tcr'] != test_tcr)
            test_mask = data_mask & (tdf['tcr'] == test_tcr)

            xtrain = fit_data[train_mask]
            xtest = fit_data[test_mask]
            ytrain = tdf.loc[train_mask, 'activation']

            # train and predict
            reg = RandomForestRegressor(
                n_jobs=n_jobs,
                n_estimators=250,
                max_features='sqrt',
                criterion='mae',
            ).fit(xtrain, ytrain)
            test_preds = reg.predict(xtest)

            # save performance
            pdf = tdf[test_mask][[
                'normalization', 'tcr', 'mut_pos', 'mut_ami',
                'wild_activation', 'activation',
            ]]
            pdf['pred'] = test_preds
            perf.append(pdf)

    ppdf = pd.concat(perf)
    ppdf['err'] = ppdf['pred'] - ppdf['activation']

    return ppdf

epitope = 'VPSVWRSSL'
fname = f'results/{epitope}_tcr_stratified_regression_performance.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

# %%  compute metrics per tcr
pdf['abserr'] = np.abs(pdf['err'])
tcr_perf = pdf.groupby(['normalization', 'tcr']).apply(lambda g: pd.Series({
    'mae': g['abserr'].mean(),
    'r2': metrics.r2_score(g['activation'], g['pred']),
    'pearson': g['activation'].corr(g['pred'], method='pearson'),
    'spearman': g['activation'].corr(g['pred'], method='spearman'),
}))

print(tcr_perf)

pdf = pdf.merge(tcr_perf, left_on=['normalization', 'tcr'], right_index=True)

#%%

sdf = tcr_perf.reset_index()
sdf['is_educated'] = sdf['tcr'].str.startswith('ED')
print(sdf.groupby('is_educated')['spearman'].apply(lambda g: g.describe()))

# %% plot regression

tcr_order = pdf.groupby('tcr') \
    .agg({'activation': 'var'}) \
    .sort_values('activation').index

g = sns.lmplot(
    x='activation',
    y='pred',
    hue='mut_pos',
    col='tcr',
    col_order=tcr_order,
    col_wrap=8,
    ci=None,
    robust=True,
    sharex=True,
    sharey=True,
    palette='husl',
    height=2,
    data=pdf.query('normalization=="AS"')
)

#g.set(xlim=(-1, 90), ylim=(-1, 90))

plt.savefig(f'figures/{epitope}_tcr_stratified_activation_regression_AS.pdf', dpi=192)

#%%

q = tcr_perf.reset_index()[[
    'normalization', 'tcr', 'spearman'
]].pivot('tcr', 'normalization')
q.columns = q.columns.get_level_values(1)

sns.pairplot(q)

plt.savefig(f'figures/{epitope}_tcr_stratified_spearman_by_norm.pdf', dpi=192)

