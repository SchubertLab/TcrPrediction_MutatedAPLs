#!/usr/bin/env python
# -*- coding: utf-8 -*-
# runs regression specific to each TCR separately model with increasingly small datasets

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm

from preprocessing import full_aa_features, get_aa_features, get_dataset


def tcr_specific_model(
    data,
    train_data,
    split_fn,
    experiment_name: str,
):
    print('running experiment', experiment_name)

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    perf = []
    for n in tqdm(data['normalization'].unique(), ncols=50):
        for t in tqdm(data['tcr'].unique(), ncols=50):
            fit_mask = (data['normalization'] == n) & (data['tcr'] == t)
            fit_data = train_data[fit_mask]

            split = split_fn(fit_data)
            for i, (train_idx, test_idx) in enumerate(split):
                xtrain = fit_data.iloc[train_idx]
                xtest = fit_data.iloc[test_idx]

                ytrain = data.loc[fit_mask, 'residual'].iloc[train_idx]

                clf = RandomForestRegressor(
                    n_jobs=n_jobs,
                    n_estimators=250,
                    max_features='sqrt',
                    criterion='mae',
                ).fit(xtrain, ytrain)
                test_preds = clf.predict(xtest)

                # save performance
                pdf = data[fit_mask].iloc[test_idx][[
                    'tcr', 'normalization', 'mut_pos', 'mut_ami',
                    'residual', 'wild_activation'
                ]]
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


fname = 'results/tcr_specific_regression_performance.csv.gz'
if not os.path.exists(fname):
    print('computing results for the first time')
    data = get_dataset(normalization='AS').query('mut_pos >= 0')
    aa_features = get_aa_features()
    train_data = full_aa_features(data, aa_features)

    ppdf = pd.concat([
        tcr_specific_model(
            data, train_data,
            split_leave_r_out(test_size=r / 100, n_splits=10),
            experiment_name=f'l{r}o'
        )
        for r in [95, 90, 75, 50, 25, 10]
    ] + [
        tcr_specific_model(data, train_data, split_leave_amino_out,
                           experiment_name='lao'),
        tcr_specific_model(data, train_data, split_leave_position_out,
                           experiment_name='lpo'),
        tcr_specific_model(data, train_data, split_leave_one_out,
                           experiment_name='lmo'),
    ])

    ppdf.to_csv(fname, index=False)
else:
    print('using cached results')
    ppdf = pd.read_csv(fname)

# %% compute metrics

def compute_metrics(g):
    return pd.Series({
        'MAE': g['abserr'].mean(),
        'R2': metrics.r2_score(g['act'], g['pred']),
        'Pearson': g['act'].corr(g['pred'], method='pearson'),
        'Spearman': g['act'].corr(g['pred'], method='spearman'),
    })


# compute metric for each validation fold separately
mdf = pd.concat([
    # except for lmo CV where each validation fold contained a single sample
    # in that case we just compute a global average for each tcr
    ppdf.query('features=="lmo"') \
        .groupby(['features', 'normalization', 'tcr']) \
        .apply(compute_metrics).reset_index(),
    ppdf.query('features!="lmo"') \
        .groupby(['features', 'normalization', 'tcr', 'fold']) \
        .apply(compute_metrics).reset_index(),
])

mdf['features'] = mdf['features'].str.upper()

lmdf = mdf.melt(
    id_vars=['tcr', 'features', 'normalization', 'fold'],
    value_vars=['R2', 'Pearson', 'Spearman', 'MAE'],
    var_name='Metric'
).rename(
    columns={'features': 'Split', 'value': 'Value'}
)

lmdf['Is Educated'] = np.where(lmdf['tcr'].str.startswith('ED'), 'Yes', 'No')
ppdf['Is Educated'] = np.where(ppdf['tcr'].str.startswith('ED'), 'Yes', 'No')

# print('average metrics on validation folds by tcr and features')
# print(lmdf.groupby([
#     'tcr', 'Split', 'normalization', 'Metric'
# ]).agg({'Value': 'mean'}).reset_index().pivot(
#     index=['tcr', 'normalization', 'Split'],
#     columns='Metric',
#     values='Value'
# ))


print('Spearman values')
print(lmdf.query(
    'normalization == "AS" & Metric == "Spearman" & Split == "LMO"'
).groupby('Is Educated')['Value'].apply(lambda g: g.describe()))

#%%

print(lmdf.query(
    'normalization == "AS" & Metric == "Spearman"'
).groupby(['Is Educated', 'Split']).agg({'Value': ['median', 'std']}))

# %% plot metrics for all tcrs together by split

g = sns.catplot(
    data=lmdf.query('normalization == "AS"'),
    x='Split', y='Value', col='Metric', row='normalization',
    hue='Is Educated',
    sharey=False, #order=[o.upper() for o in order], kind='box',
    ci='sd', height=3, margin_titles=True,
)

for k, ax in g.axes_dict.items():
    ax.tick_params(axis='x', rotation=90)
    if k[1] != 'MAE':
        ax.set_ylim(-0.6, 1.1)
g.set_titles(col_template="{col_name}")

g.savefig('figures/validation-metrics-by-split-together.pdf', dpi=192)
g.savefig('figures/validation-metrics-by-split-together.png', dpi=300)

# %% plot spearman for all TCRs by split

g = sns.catplot(
    data=lmdf.query('Metric=="Spearman"'),
    x='Split', y='Value', col='Metric', hue='Is Educated',
    hue_order=['Yes', 'No'],
    sharey=False, #order=[o.upper() for o in order], kind='box',
    ci='sd', height=3.5, aspect=1.25,
)
for ax in g.axes_dict.values():
    ax.tick_params(axis='x', rotation=90)
g.set_titles(col_template="{col_name}")

g.savefig('figures/validation-spearman-by-split-together.pdf', dpi=192)
g.savefig('figures/validation-spearman-by-split-together.png', dpi=300)


# %% compare lmo metrics across tcrs

order = lmdf.query(
    'Split=="LMO" & Metric=="Spearman" & normalization == "AS"'
).sort_values('Value')['tcr']

g = sns.catplot(
    data=lmdf.query('Split=="LMO" & normalization == "AS"'),
    x='Value', y='tcr', col='Metric',
    order=order, row='normalization',
    sharex=False, height=4, aspect=0.7,
    margin_titles=True,
)
for k, ax in g.axes_dict.items():
    #ax.tick_params(rotation=90)
    #if k[1] == 'AS':
    #    ax.set_xlim(0, 1)
    pass

g.savefig('figures/validation-metrics-by-tcr.pdf', dpi=192)

# %% find which amino acids are harder to predict

cc = ppdf.query('features=="lao" & normalization == "AS"') \
    .groupby(['tcr', 'Is Educated', 'normalization', 'mut_ami']) \
    .apply(compute_metrics) \
    .reset_index() \
    .melt(['tcr', 'Is Educated', 'normalization', 'mut_ami'])

order = cc.query('variable=="Spearman"') \
    .groupby('mut_ami') \
    .agg({'value': 'mean'}) \
    .sort_values('value') \
    .index.to_list()

g = sns.catplot(
    data=cc,
    x='mut_ami',
    y='value',
    col='variable',
    row='normalization',
    kind='box',
    hue='Is Educated',
    dodge=True,
    margin_titles=True,
    sharey=False,
    order=order,
    height=3.5
)
g.axes_dict['AS', 'R2'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Pearson'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Spearman'].set(ylim=(-0.25, 1))
plt.savefig('figures/metrics-by-left-out-amino.pdf', dpi=192)

# %%

dd = cc.merge(
    ppdf.query('features=="lao"')
        .groupby(['tcr', 'mut_ami'])
        .agg({'act': 'std'})
        .rename(columns={'act': 'activation_std'})
        .reset_index(),
    on=['tcr', 'mut_ami']
)

g = sns.FacetGrid(
    data=dd,
    col='variable',
    sharey=False,
    hue='Is Educated',
    height=3.5,
    row='normalization',
)
g.map(sns.scatterplot, 'activation_std', 'value')#, 'mut_ami')
g.axes_dict['AS', 'R2'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Pearson'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Spearman'].set(ylim=(-0.25, 1))
g.add_legend()
plt.savefig('figures/left-out-amino-metric-vs-activation-std.pdf', dpi=192)

# %%
print(
    ppdf.query('features == "lao" & normalization == "AS"') \
        .groupby(['tcr', 'mut_ami']) \
        .agg({'act': 'var'}) \
        .rename(columns={'act': 'activation_variance'}) \
        .reset_index().sort_values('activation_variance')
)

# %% find which positions are harder to predict

cc = ppdf.query('features=="lpo" & normalization == "AS"') \
    .groupby(['tcr', 'Is Educated', 'normalization', 'mut_pos']) \
    .apply(compute_metrics) \
    .reset_index() \
    .melt(['tcr', 'Is Educated', 'normalization', 'mut_pos'])


order = cc.query('variable=="Spearman"') \
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
    row='normalization',
    hue='Is Educated',
    sharey=False,
    margin_titles=True,
    order=order,
    height=3.5
)

g.axes_dict['AS', 'R2'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Pearson'].set(ylim=(-0.25, 1))
g.axes_dict['AS', 'Spearman'].set(ylim=(-0.25, 1))
plt.savefig('figures/metrics-by-left-out-position.pdf', dpi=192)

# %% regression lines for all lmo features and all tcrs
order = ppdf.groupby('tcr') \
            .agg({'act': 'var'}) \
            .sort_values('act').index

g = sns.lmplot(
    data=ppdf.query('features=="lmo" & normalization == "AS"'),
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
#g.set(xlim=(0, 80), ylim=(0, 80))
plt.savefig('figures/tcr_specific_regression_lmo_features.pdf', dpi=192)
