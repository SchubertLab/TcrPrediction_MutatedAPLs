#!/usr/bin/env python
# -*- coding: utf-8 -*-
# runs regression specific to each TCR separately model with increasingly small datasets

import os
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from preprocessing import get_dataset, get_aa_features, full_aa_features

#%% load data

df = get_dataset()
aa_features = get_aa_features()
data = df[(
    df['mut_pos'] >= 0
) & (
     df['tcr'].isin(df.query('activation > 15')['tcr'].unique())
)]
train_data = full_aa_features(data, aa_features)

#%% evaluation on different split strategies

def tcr_specific_model(
    split_fn,
    experiment_name: str,
):
    print('running experiment', experiment_name)
    perf = []
    for t in tqdm(data['tcr'].unique()):
        fit_mask = (data['tcr'] == t)
        fit_data = train_data[fit_mask]

        split = split_fn(fit_data)
        for i, (train_idx, test_idx) in enumerate(split):
            xtrain = fit_data.iloc[train_idx]
            xtest = fit_data.iloc[test_idx]

            ytrain = data.loc[fit_mask, 'residual'].iloc[train_idx]

            clf = RandomForestRegressor().fit(xtrain, ytrain)
            test_preds = clf.predict(xtest)

            # save performance
            pdf = data[fit_mask].iloc[test_idx][[
                'mut_pos', 'mut_ami', 'residual', 'wild_activation'
            ]]
            pdf['tcr'] = t
            pdf['fold'] = i
            pdf['pred_res'] = test_preds
            perf.append(pdf)

    # aggregate performance data
    pdf = pd.concat(perf)

    pdf['pred'] = pdf['pred_res'] + pdf['wild_activation']
    pdf['act'] = pdf['residual'] + pdf['wild_activation']

    pdf['abserr'] = np.abs(pdf['residual'] - pdf['pred_res'])
    pdf['err'] = pdf['pred_res'] - pdf['residual']

    if experiment_name:
        pdf['features'] = experiment_name

    return pdf


def split_leave_one_out(data):
    yield from KFold(len(data), shuffle=True).split(data)


def split_leave_position_out(data):
    for p in range(8):
        train_idx = np.flatnonzero(data['mut_pos'] != p)
        vali_idx = np.flatnonzero(data['mut_pos'] == p)
        yield train_idx, vali_idx


def split_leave_amino_out(data):
    for a in 'ACDEFGHIKLMNPQRSTVWY':
        train_idx = np.flatnonzero(data[f'mut_ami$one_hot$is_{a}'] < 0.5)
        vali_idx = np.flatnonzero(data[f'mut_ami$one_hot$is_{a}'] > 0.5)
        yield train_idx, vali_idx


def split_leave_r_out(test_size, n_splits):
    def split(data):
        yield from ShuffleSplit(n_splits, test_size=test_size).split(data)
    return split


fname = 'results/tcr_specific_performance.csv'
if not os.path.exists(fname):
    print('computing results for the first time')
    ppdf = pd.concat([
        tcr_specific_model(
            split_leave_r_out(test_size=r / 100, n_splits=10),
            experiment_name=f'l{r}o'
        )
        for r in [95, 90, 75, 50, 25, 10]
    ] + [
        tcr_specific_model(split_leave_amino_out, experiment_name='lao'),
        tcr_specific_model(split_leave_position_out, experiment_name='lpo'),
        tcr_specific_model(split_leave_one_out, experiment_name='lmo'),
    ])

    ppdf.to_csv(fname, index=False)
else:
    print('using cached results')
    ppdf = pd.read_csv(fname)

#%% compute metrics

def compute_metrics(g):
    return pd.Series({
        'mae': g['abserr'].mean(),
        'r2': metrics.r2_score(g['act'], g['pred']),
        'pearson': g['act'].corr(g['pred'], method='pearson'),
        'spearman': g['act'].corr(g['pred'], method='spearman'),
    })


# compute metric for each validation fold separately
mdf = pd.concat([
    # except for lmo CV where each validation fold contained a single sample
    # in that case we just compute a global average for each tcr
    ppdf.query('features=="lmo"') \
        .groupby(['features', 'tcr']) \
        .apply(compute_metrics).reset_index(),
    ppdf.query('features!="lmo"') \
        .groupby(['features', 'tcr', 'fold']) \
        .apply(compute_metrics).reset_index(),
])

lmdf = mdf.melt(
    id_vars=['tcr', 'features', 'fold'],
    value_vars=['r2', 'pearson', 'spearman', 'mae'],
    var_name='metric'
)

print('average metrics on validation folds by tcr and features')
print(lmdf.groupby([
    'tcr', 'features', 'metric'
]).agg({'value': 'mean'}).reset_index().pivot(
    index=['tcr', 'features'],
    columns='metric',
    values='value'
))

#%% plot spearman for each tcr separately

order =['lmo', 'l10o', 'l25o', 'l50o', 'lao', 'l75o', 'l90o', 'l95o', 'lpo']

g = sns.catplot(
    data=lmdf.query('metric=="spearman"'),
    x='features', y='value', col='tcr', col_wrap=7,
    sharey=True, order=order, kind='strip',
)
g.map(sns.pointplot, 'features', 'value', order=order, color='gray')
g.savefig('figures/spearman-by-split.pdf', dpi=192)

#%% plot metrics for all tcrs together

g = sns.catplot(
    data=lmdf,
    x='features', y='value', col='metric',
    sharey=False, order=order, kind='box',
    ci='sd', height=3.5,
    #hue='tcr',
)
g.axes_dict['r2'].set_ylim(0, 1)
g.savefig('figures/validation-metrics-by-split-together.pdf', dpi=192)

#%% find which amino acids are harder to predict

cc = ppdf.query('features=="lao"') \
        .groupby(['tcr', 'mut_ami']) \
        .apply(compute_metrics) \
        .reset_index() \
        .melt(['tcr', 'mut_ami'])

order = cc.query('variable=="spearman"') \
        .groupby('mut_ami') \
        .agg({'value': 'mean'}) \
        .sort_values('value') \
        .index.to_list()

g = sns.catplot(
    data=cc,
    x='mut_ami',
    y='value',
    col='variable',
    kind='box',
    dodge=True,
    sharey=False,
    order=order,
    height=3.5
)
g.axes_dict['r2'].set(ylim=(0, 1))
plt.savefig('figures/metrics-by-left-out-amino.pdf', dpi=192)

#%% find which positions are harder to predict

cc = ppdf.query('features=="lpo"') \
        .groupby(['tcr', 'mut_pos']) \
        .apply(compute_metrics) \
        .reset_index() \
        .melt(['tcr', 'mut_pos'])

order = cc.query('variable=="spearman"') \
            .groupby('mut_pos') \
            .agg({'value': 'mean'}) \
            .sort_values('value') \
            .index.to_list()


g = sns.catplot(
    data=cc,
    x='mut_pos',
    y='value',
    kind='box',
    dodge=True,
    col='variable',
    sharey=False,
    order=order,
    height=3.5
)

g.axes_dict['r2'].set(ylim=(0, 1))
plt.savefig('figures/metrics-by-left-out-position.pdf', dpi=192)

#%% regression lines for all lmo features and all tcrs
order = data.groupby('tcr') \
            .agg({'activation': 'var'}) \
            .sort_values('activation').index

g = sns.lmplot(
    data=ppdf.query('features=="lmo"'),
    x='act',
    y='pred',
    hue='mut_pos',
    col='tcr',
    col_wrap=8,
    ci=None,
    robust=True,
    sharex=True,
    sharey=True,
    palette='husl',
    height=2,
    col_order=order
)
g.set(xlim=(0, 80), ylim=(0, 80))
plt.savefig('figures/tcr_specific_regression_lmo_features.pdf', dpi=192)
