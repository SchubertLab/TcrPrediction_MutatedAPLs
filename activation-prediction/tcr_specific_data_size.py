#!/usr/bin/env python
# -*- coding: utf-8 -*-
# runs regression and classification TCR-specific models with increasingly small datasets

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm

from preprocessing import full_aa_features, get_aa_features, get_dataset, get_tumor_dataset


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

                # regression
                yres = data.loc[fit_mask, 'residual'].iloc[train_idx]
                rfreg = RandomForestRegressor(
                    n_jobs=n_jobs,
                    n_estimators=250,
                    max_features='sqrt',
                    criterion='mae',
                ).fit(xtrain, yres)
                res_pred = rfreg.predict(xtest)
                
                # classification
                yact = data.loc[fit_mask, 'is_activated'].iloc[train_idx]
                act_pred = None
                if yact.min() != yact.max():
                    rfcls = RandomForestClassifier(
                        n_jobs=n_jobs,
                        n_estimators=250,
                        max_features='sqrt',
                    ).fit(xtrain, yact)
                    act_pred = rfcls.predict_proba(xtest)[:, 1]

                # save performance
                pdf = data[fit_mask].iloc[test_idx][[
                    'tcr', 'normalization', 'mut_pos', 'mut_ami',
                    'residual', 'wild_activation', 'is_activated'
                ]]
                pdf['fold'] = i
                pdf['pred_res'] = res_pred
                pdf['pred_prob'] = act_pred
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


epitope = 'VPSVWRSSL'
fname = f'results/{epitope}_tcr_specific_data_size.csv.gz'
if not os.path.exists(fname):
    print('computing results for the first time')
    if epitope == 'VPSVWRSSL':
        data = get_tumor_dataset()
    else:
        data = get_dataset(normalization='AS')
    data = data.query('mut_pos >= 0')
    data['is_activated'] = data['activation'] > 46.9
    data = data[(
        data['mut_pos'] >= 0
    ) & data['tcr'].isin(
        set(
            data.query('is_activated')['tcr'].unique()
        ) & set(
            data.query('~is_activated')['tcr'].unique()
        )
    )]
    
    aa_features = get_aa_features()
    train_data = full_aa_features(data, aa_features[['factors']], base_peptide=epitope)

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
        'AUC': (
            metrics.roc_auc_score(g['is_activated'], g['pred_prob'])
            if np.isfinite(g['pred_prob']).all() and 0 < g['is_activated'].mean() < 1
            else np.nan
        ),
        'APS': (
            metrics.average_precision_score(g['is_activated'], g['pred_prob'])
            if np.isfinite(g['pred_prob']).all()
            else np.nan
        ),
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
    value_vars=['R2', 'Pearson', 'Spearman', 'MAE', 'APS', 'AUC'],
    var_name='Metric'
).rename(
    columns={'features': 'Split', 'value': 'Value'}
)

lmdf['Repertoire'] = np.where(lmdf['tcr'].str.startswith('ED'), 'Educated', 'Naive')
ppdf['Repertoire'] = np.where(ppdf['tcr'].str.startswith('ED'), 'Educated', 'Naive')

# %% plot spearman for all TCRs by split
g = sns.catplot(
    data=lmdf.query('Metric=="Spearman"'),
    x='Split', y='Value', col='Metric', hue='Repertoire',
    kind='box',
    sharey=False, #order=[o.upper() for o in order], kind='box',
    ci='sd', height=3, aspect=1.25,
    order=['LMO', 'L10O', 'L25O', 'L50O', 'LAO', 'L75O', 'L90O', 'L95O', 'LPO'],
    legend=False,
)
for ax in g.axes_dict.values():
    ax.tick_params(axis='x', rotation=90)
g.set_titles(col_template="{col_name}")
g.add_legend(loc='lower left')
g.savefig(f'figures/{epitope}_validation-spearman-by-split-together.pdf', dpi=192)
g.savefig(f'figures/{epitope}_validation-spearman-by-split-together.png', dpi=300)


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

g.savefig(f'figures/{epitope}_validation-metrics-by-tcr.pdf', dpi=192)

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
plt.savefig(f'figures/{epitope}_metrics-by-left-out-amino.pdf', dpi=192)

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
plt.savefig(f'figures/{epitope}_left-out-amino-metric-vs-activation-std.pdf', dpi=192)

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
    data=cc[cc['Is Educated'] == 'Yes'].query('variable=="AUC"'),
    x='mut_pos',
    y='value',
    kind='box',
    dodge=True,
    #col='variable',
    #row='normalization',
    #hue='Is Educated',
    sharey=False,
    #margin_titles=True,
    order=range(8),
    height=3.5,
    palette='husl',
    zorder=2,
    showmeans=True,
    notch=True,
    meanprops={'mfc': 'k', 'mec': 'k'}
)

g.set(xticklabels=[f'P{i+1}' for i in range(8)],
      xlabel='Validate on position',
      ylabel='AUC for educated repertoire')
#g.axes_dict['AS', 'R2'].set(ylim=(-0.25, 1))
#g.axes_dict['AS', 'Pearson'].set(ylim=(-0.25, 1))
#g.axes_dict['AS', 'Spearman'].set(ylim=(-0.25, 1))
plt.tight_layout()
plt.savefig(f'figures/{epitope}_metrics-by-left-out-position.pdf', dpi=300,
            bbox_inches='tight')

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
plt.savefig(f'figures/{epitope}_tcr_specific_regression_lmo_features.pdf', dpi=192)
