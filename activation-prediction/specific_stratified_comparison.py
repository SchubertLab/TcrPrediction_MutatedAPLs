#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm

from preprocessing import full_aa_features, get_aa_features, get_dataset

#%% load data

reg_strat_df = pd.read_csv('results/tcr_stratified_regression_performance.csv.gz')
reg_spec_df = pd.read_csv('results/tcr_specific_regression_performance.csv.gz') \
                .query('features=="lmo"') \
                .rename(columns={'act': 'activation'})

cls_strat_df = pd.read_csv('results/tcr_stratified_classification_performance.csv.gz')
cls_spec_df = pd.read_csv('results/tcr_specific_classification_performance.csv.gz')

# keep common tcrs
reg_tcrs = set(reg_spec_df['tcr'].unique()) \
            & set(reg_strat_df['tcr'].unique()) \
            & set(cls_strat_df['tcr'].unique()) \
            & set(cls_spec_df['tcr'].unique())

reg_strat_df = reg_strat_df[reg_strat_df['tcr'].isin(reg_tcrs)].query('normalization=="AS"')
reg_spec_df = reg_spec_df[reg_spec_df['tcr'].isin(reg_tcrs)].query('normalization=="AS"')
cls_spec_df = cls_spec_df[
    cls_spec_df['tcr'].isin(reg_tcrs)
].query('normalization=="AS" & threshold == 46.9')
cls_strat_df = cls_strat_df[
    cls_strat_df['tcr'].isin(reg_tcrs)
].query('normalization=="AS" & threshold == 46.9')

#%%

def regression_metrics(g):
    return pd.Series({
        'MAE': np.mean(np.abs(g['activation'] - g['pred'])),
        'R2': metrics.r2_score(g['activation'], g['pred']),
        'Pearson': g['activation'].corr(g['pred'], method='pearson'),
        'Spearman': g['activation'].corr(g['pred'], method='spearman'),
    })


def classification_metrics(g):
    best_mcc = best_f1 = 0.0
    for thr in np.linspace(0, 1, 20):
        best_mcc = max(
            best_mcc,
            metrics.matthews_corrcoef(g['is_activated'], g['pred'] > thr)
        )
        best_f1 = max(
            best_f1,
            metrics.f1_score(g['is_activated'], g['pred'] > thr)
        )

    return pd.Series({
        'AUC': metrics.roc_auc_score(g['is_activated'], g['pred']),
        'APS': metrics.roc_auc_score(g['is_activated'], g['pred']),
        'MCC': best_mcc,
        'F1': best_f1,
    })


reg_spec_metrics = reg_spec_df.groupby('tcr') \
    .apply(regression_metrics) \
    .reset_index() \
    .melt('tcr', var_name='metric', value_name='tcr-specific') \
    .set_index(['tcr', 'metric'])

reg_strat_metrics = reg_strat_df.groupby('tcr') \
    .apply(regression_metrics) \
    .reset_index() \
    .melt('tcr', var_name='metric', value_name='tcr-stratified') \
    .set_index(['tcr', 'metric'])

cls_spec_metrics = cls_spec_df.groupby('tcr') \
    .apply(classification_metrics) \
    .reset_index() \
    .melt('tcr', var_name='metric', value_name='tcr-specific') \
    .set_index(['tcr', 'metric'])

cls_strat_metrics = cls_strat_df.groupby('tcr') \
    .apply(classification_metrics) \
    .reset_index() \
    .melt('tcr', var_name='metric', value_name='tcr-stratified') \
    .set_index(['tcr', 'metric'])


metrics_df = pd.concat([
    reg_spec_metrics.merge(
        reg_strat_metrics, left_index=True, right_index=True
    ).reset_index(),
    cls_spec_metrics.merge(
        cls_strat_metrics, left_index=True, right_index=True
    ).reset_index(),
]).rename(columns={
    'metric': 'Metric',
    'tcr-specific': 'TCR-specific',
    'tcr-stratified': 'TCR-stratified',
    'tcr': 'TCR',
})

metrics_df['is_educated'] = metrics_df['TCR'].str.startswith('ED')

#%%
g = sns.FacetGrid(data=metrics_df, col='Metric', col_wrap=4,
                  height=3, sharey=False, sharex=False)
g.map(sns.scatterplot, 'TCR-specific', 'TCR-stratified', 'is_educated')
g.axes_dict['R2'].set(xlim=(0, 1), ylim=(0, 1))
# add line on diagonal
for ax in g.axes_dict.values():
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lb = min(xmin, ymin)
    ub = max(xmax, ymax)
    ax.plot([lb, ub], [lb, ub], 'r--')


# from adjustText import adjust_text
# def annot(x, y, color, data):
#     err = np.abs(data[x] - data[y])
#     t = sorted(err.values)[-5]
#     mask = err >= t

#     xl, xu = plt.xlim()
#     yl, yu = plt.ylim()

#     txt = [
#         plt.text(max(min(x, xu), xl),
#                  max(min(y, yu), yl),
#                  tcr)
#         for tcr, x, y in data.loc[mask, ['TCR', x, y]].values
#     ]
#     adjust_text(txt, arrowprops=dict(arrowstyle='-'))

# g.map_dataframe(annot, 'TCR-specific', 'TCR-stratified')

g.add_legend(title='Educated')
#g.tight_layout()
g.set_titles(col_template="{col_name}")
g.savefig('figures/all-metrics-specific-vs-stratified.pdf', dpi=192)
g.savefig('figures/all-metrics-specific-vs-stratified.png', dpi=300)
