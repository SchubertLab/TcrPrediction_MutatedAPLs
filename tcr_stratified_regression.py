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


def train():
    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    tdf = df.query('mut_pos>=0')
    fit_data = full_aa_features(tdf, aa_features, interactions=True, include_tcr=True)

    print('total features', fit_data.shape[1])

    perf = []
    #for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
    for test_tcr in tqdm(['B11', 'B15', 'OT1']):
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


#%% training and evaluation

if not os.path.exists('regression_performance.csv'):
    df = get_dataset()

    # works a bit better by stratifying
    df = df[df['tcr'].isin({'F4', 'B13', 'B11', 'OT1', 'B15'})]

    aa_features = get_aa_features()
    qq = train()
    qq.to_csv('regression_performance.csv', index=False)
else:
    print('using cached results')

#%% read evaluation data

pdf = pd.read_csv('regression_performance.csv')
pdf = pdf[pdf['tcr'].isin({'OT1', 'B11', 'B15', 'F4', 'B13', 'G6'})]

# compute absolute error per tcr
pdf['abserr'] = np.abs(pdf['err'])
tcr_maes = pdf.groupby('tcr').apply(lambda g: pd.Series({
    'tcr_mae': g['abserr'].mean(),
    'r2': metrics.r2_score(g['activation'], g['pred']),
    'pearson': g['activation'].corr(g['pred'], method='pearson'),
    'spearman': g['activation'].corr(g['pred'], method='spearman'),
}))

#        tcr_mae        r2   pearson  spearman
# tcr
# B11   8.837565  0.196002  0.848980  0.863708
# B15  16.432858 -0.577070  0.712094  0.724241
# OT1   9.576967  0.051483  0.864085  0.869605

print(tcr_maes)

pdf = pdf.merge(tcr_maes, left_on='tcr', right_index=True)

#%% plot regression

pdf['tcr_title'] = pdf.apply(
    lambda row: '{} (MAE: {:.4f})'.format(row['tcr'], row['tcr_mae']),
    axis=1
)

g = sns.lmplot(
    x='activation',
    y='pred',
    hue='mut_pos',
    col='tcr_title',
    col_wrap=3,
    ci=None, robust=True,
    sharex=True, sharey=True,
    palette='husl',
    data=pdf
)

g.set(xlim=(-1, 50), ylim=(-1, 50))

plt.savefig('figures/tcr_activation_regression.pdf', dpi=192)
