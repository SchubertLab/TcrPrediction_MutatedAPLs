#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this trains a random forest validating on a held-out receptor


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from preprocessing import get_dataset, get_aa_features, full_aa_features
import os
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt


#%% training and evaluation

def train(tdf, aa_features):
    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)
    print('total features', fit_data.shape[1])

    perf = []
    for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
        test_mask = tdf['tcr'] == test_tcr

        xtrain = fit_data[~test_mask]
        xtest = fit_data[test_mask]
        ytrain = tdf.loc[~test_mask, 'activation']

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
            'tcr', 'mut_pos', 'mut_ami', 'wild_activation', 'activation',
        ]]
        pdf['pred'] = test_preds
        perf.append(pdf)

    ppdf = pd.concat(perf)
    ppdf['err'] = ppdf['pred'] - ppdf['activation']

    return ppdf


fname = 'results/tcr_stratified_regression_performance.csv'
if not os.path.exists(fname):
    df = get_dataset()
    data = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    ) & (
        df['tcr'].isin(df.query('activation > 15')['tcr'].unique())
    )]

    aa_features = get_aa_features()
    pdf = train(data, aa_features)
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

#%%  compute metrics per tcr
pdf['abserr'] = np.abs(pdf['err'])
tcr_maes = pdf.groupby('tcr').apply(lambda g: pd.Series({
    'tcr_mae': g['abserr'].mean(),
    'r2': metrics.r2_score(g['activation'], g['pred']),
    'pearson': g['activation'].corr(g['pred'], method='pearson'),
    'spearman': g['activation'].corr(g['pred'], method='spearman'),
}))

print(tcr_maes)

pdf = pdf.merge(tcr_maes, left_on='tcr', right_index=True)

#%% plot regression

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
    data=pdf
)

g.set(xlim=(-1, 90), ylim=(-1, 90))

plt.savefig('figures/tcr_stratified_activation_regression.pdf', dpi=192)
