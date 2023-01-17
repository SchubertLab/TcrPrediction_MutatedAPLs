#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script estimates the permutation feature importance for tcr-stratified classification

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

from preprocessing import (add_activation_thresholds, build_feature_groups,
                           decorrelate_groups, full_aa_features,
                           get_aa_factors, get_aa_features,
                           get_complete_dataset, get_dataset, get_tumor_dataset)


#%% training and evaluation
def shuffle(df, col_mask, keep_rows=None):
    if keep_rows is None:
        keep_rows = np.array([True] * len(df))

    assert len(df.index) == len(set(df.index)), 'index must be unique'

    xgroup = df[keep_rows].loc[:, col_mask]
    xothers = df[keep_rows].loc[:, ~col_mask]

    idx = np.random.permutation(xgroup.index)

    xshuffle = pd.concat([
        xgroup.loc[idx].reset_index(drop=True),
        xothers.reset_index(drop=True),
    ], axis=1)
    assert np.all(np.isfinite(xshuffle)), 'failed to concat'

    xshuffle = xshuffle[df.columns]  # re-arrange columns
    assert (col_mask.sum() == 0
            or np.all(df.loc[keep_rows, col_mask].std(axis=0) < 1e-6)
            or np.any(xshuffle.values != df[keep_rows].values)), 'failed to shuffle'

    xshuffle.index = df.index
    return xshuffle


def train():
    df = get_dataset(normalization='AS') if epitope=='SIINFEKL' else get_tumor_dataset()
    df = df[df['tcr'].isin(['R24', 'R28'])]

    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    )]
    tdf['is_activated'] = tdf['activation'] > default_threshold

    aa_features = get_aa_features()
    fit_data = full_aa_features(tdf, aa_features[['factors']], include_tcr=True, base_peptide=epitope)

    # remove position, original and mutated amino acid
    # so that the model only relies on sequence information
    fit_data = fit_data[[
        c for c in fit_data.columns
        if 'orig_ami' not in c and 'mut_ami' not in c and 'mut_pos' not in c
    ]]

    feature_groups = build_feature_groups(fit_data.columns, length=len(epitope))

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    perf = []
    for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
        test_mask = tdf['tcr'] == test_tcr

        xtrain = fit_data.loc[~test_mask]
        ytrain = tdf.loc[~test_mask, 'is_activated']

        # train and predict
        reg = RandomForestClassifier(
            n_estimators=1000,
            n_jobs=n_jobs,
        ).fit(xtrain, ytrain)

        for group, cols in tqdm(feature_groups.items(), ncols=50, leave=False):
            column_mask = np.array([
                any(c.startswith(d) for d in cols)
                for c in fit_data.columns
            ])
            if group != 'all' and column_mask.sum() == 0:
                continue

            if group.startswith('cdr3a_') or group.startswith('cdr3b_'):
                # do not shuffle individual cdr3a/b positions
                # given the high collinearity between them I don't think
                # we can get reliable information about the importance of
                # a single position there
                continue

            # perform successive rounds of shuffling and evaluation
            # since we are performing a leave-tcr-out evaluation, shuffling includes
            # samples from tcrs in the training set
            for i in range(15):
                xshuffle = shuffle(fit_data, column_mask)
                shuffle_preds = reg.predict_proba(xshuffle[test_mask])[:, 1]

                pdf = tdf[test_mask][[
                    'tcr', 'mut_pos', 'mut_ami', 'wild_activation',
                    'activation', 'is_activated'
                ]]
                pdf['pred'] = shuffle_preds
                pdf['group'] = group
                pdf['shuffle'] = i
                perf.append(pdf)

    ppdf = pd.concat(perf)

    return ppdf

parser = argparse.ArgumentParser()
parser.add_argument('--epitope', type=str, default='SIINFEKL')
parser.add_argument('--activation', type=str, default='AS')
parser.add_argument('--threshold', type=float, default=46.9)
params = parser.parse_args()

epitope = params.epitope
default_activation = params.activation
default_threshold = params.threshold

#%%
fname = f'results/{epitope}_tcr_stratified_permutation_importance.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)


#%% computing performance (only for activated tcrs)
mdf = pdf[pdf['tcr'].isin(
    pdf[pdf['is_activated'] > 0.5]['tcr'].unique()
)].groupby(['tcr', 'group', 'shuffle']).apply(lambda q: pd.Series({
    'auc': roc_auc_score(q['is_activated'], q['pred']),
    'aps': average_precision_score(q['is_activated'], q['pred']),
    #'spearman': stats.spearmanr(q['activation'], q['pred'])[0],
})).reset_index().drop(columns='shuffle')


#%%
ddf = mdf.melt(['tcr', 'group']).merge(
    mdf[
        mdf['group'] == 'all'
    ].drop(columns='group').melt('tcr', value_name='base').drop_duplicates(),
    on=['tcr', 'variable']
)
ddf['diff'] = ddf['value'] - ddf['base']
ddf['rel'] = ddf['value'] / ddf['base'] - 1  # positive = increase
ddf['item'] = ddf['group'].str.split('_').str[0]
ddf['is_educated'] = np.where(
    ddf['tcr'].str.startswith('ED') | ddf['tcr'].str.startswith('R') ,
    'Educated', 'Naive'
)

#%%
print(ddf.query(
    'variable == "auc" & is_educated'
).groupby('group')['value'].median().sort_values())

#%% feature importance for educated
g = sns.catplot(
    data=ddf[(
        ddf['is_educated'] == "Educated"
    ) & (
        ddf['variable'] == 'auc'
    ) & (
        ddf['group'].str.startswith('pos_')
          | ddf['group'].isin(['cdr3', 'all'])
    )].rename(columns={
        'value': 'AUC', 'group': 'Permutation'
    }).replace({
        'pos_0': 'P1', 'pos_1': 'P2', 'pos_2': 'P3', 'pos_3': 'P4',
        'pos_4': 'P5', 'pos_5': 'P6', 'pos_6': 'P7', 'pos_7': 'P8', 'pos_8': 'P9',
        'cdr3': 'CDR3', 'all': '-'
    }),
    x='Permutation',
    y='AUC',
    sharey=True,
    #ci='sd',
    #dodge=True,
    aspect=1.25,
    height=3,
    kind='box',
    #hue='tcr',
    palette='husl',
    zorder=2,
    showmeans=True,
    notch=True,
    meanprops={'mfc': 'k', 'mec': 'k'}
)

g.set(ylim=(0.5, 1), ylabel='AUC on educated repertoire',
      xlabel='Group permuted')

g.savefig(f'figures/{epitope}_permutation_feature_importance_educated.pdf', dpi=192)
g.savefig(f'figures/{epitope}_permutation_feature_importance_educated.png', dpi=192)


#%% decrease by position

pddf = ddf.query(
    'tcr!="G6" & variable=="auc" & item == "pos"'
).rename(columns={
    'rel': 'Relative Increase',
    'tcr': 'TCR',
})
pddf['Position'] = pddf['group'].apply(lambda s: f'P{int(s[-1])+1}')

g = sns.catplot(
    data=pddf,
    #data=pddf.groupby(['TCR', 'group']).apply(lambda g: g.head(2)).reset_index(drop=True),
    x='Position',
    y='Relative Increase',
    dodge=True,
    aspect=1.4,
    height=4,
    kind='point',
    ci='sd',
    legend=False,
    #order=[f'P{i}' for i in range(1, 9)],
    order=pddf.groupby(['Position']).agg({
        'Relative Increase': 'mean'
    }).sort_values('Relative Increase').index,
    zorder=2,  # above points
)

g.map(sns.stripplot,
      'Position',
      'Relative Increase',
      'TCR',
      dodge=True,
      #order=[f'P{i}' for i in range(1, 9)],
      order=pddf.groupby(['Position']).agg({'Relative Increase': 'mean'}).sort_values('Relative Increase').index,
      hue_order=sorted(pddf['TCR'].unique()),
      palette='husl',
      zorder=1
)
g.add_legend(title='TCR', ncol=2)
g.ax.set_yticklabels([f'{100 * y:.0f}%' for y in g.ax.get_yticks()])

g.savefig(f'figures/{epitope}_permutation_feature_importance_positions.pdf', dpi=192)
g.savefig(f'figures/{epitope}_permutation_feature_importance_positions.png', dpi=192)
