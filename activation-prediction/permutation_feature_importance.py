#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script estimates the permutation feature importance for tcr-stratified classification

from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from preprocessing import get_dataset, get_aa_features, full_aa_features, get_aa_factors, build_feature_groups, decorrelate_groups
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import os
from tqdm import tqdm

#%%

# from preprocessing import get_complete_dataset
# ds = get_complete_dataset()
# ps = ds.pivot_table(index=['tcr', 'mut_pos', 'mut_ami'], columns='normalization', values='activation')
# g = sns.pairplot(ps.reset_index(), vars=['AS', 'OT1', 'none'], hue='tcr')
# plt.suptitle('Comparison of activation values for different normalizations')
# plt.savefig('/tmp/norm.pdf', dpi=192)


# c3as = set(tdf['cdr3a_aligned'].tolist())
# counts = {}
# for c in c3as:
#     for i, a in enumerate(c):
#         if i not in counts:
#             counts[i] = {}
#         if a not in counts[i]:
#             counts[i][a] = 0
#         counts[i][a] += 1

#%% training and evaluation

def shuffle(df, col_mask, keep_rows=None):
    if keep_rows is None:
        keep_rows = np.array([True] * len(df))

    xgroup = df[keep_rows].loc[:, col_mask]
    xothers = df[keep_rows].loc[:, ~col_mask]

    idx = np.random.permutation(xgroup.index)

    xshuffle = pd.concat([
        xgroup.loc[idx].reset_index(drop=True),
        xothers.reset_index(drop=True),
    ], axis=1)
    assert np.all(np.isfinite(xshuffle)), 'failed to concat'

    xshuffle = xshuffle[df.columns]
    assert (col_mask.sum() == 0
            or np.all(df.loc[keep_rows, col_mask].std(axis=0) < 1e-6)
            or np.any(xshuffle.values != df[keep_rows].values)), 'failed to shuffle'

    xshuffle.index = df.index
    return xshuffle


def train():
    df = get_dataset()
    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    ) & (
        df['tcr'].isin(df.query('activation > 15')['tcr'].unique())
    )]
    tdf['is_activated'] = (tdf['activation'] > 15).astype(np.int64)

    aa_features = get_aa_features()
    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)

    # remove position, original and mutated amino acid
    # so that the model only relies on sequence information
    fit_data = fit_data[[
        c for c in fit_data.columns
        if 'orig_ami' not in c and 'mut_ami' not in c and 'mut_pos' not in c
    ]]

    feature_groups = build_feature_groups(fit_data.columns)

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
            
            # exclude samples in this group with gaps in whis position from shuffling,
            # meaning that (1) if the tcr has a gap in this particular position it will
            # be excluded from evaluation and (2) otherwise, shuffling will not introduce
            # a gap in this position.
            # this is to preserve realistic alignments without introducing or removing
            # gaps. this does not apply to shuffling the entire cdr3 or alpha/beta chains,
            # since doing so preserves biological relevance
            gaps_cols = np.array([
                any(c.startswith(d) for d in cols) and 'is_gap' in c
                for c in fit_data.columns
            ])
            
            if group not in ('cdr3', 'cdr3a', 'cdr3b') and gaps_cols.any():
                nogaps_mask = (fit_data.loc[:, gaps_cols] < 0.5).any(axis=1)
            else:
                nogaps_mask = pd.Series(np.ones(fit_data.shape[0]).astype(np.bool),
                                        index=fit_data.index)

            if not np.any(test_mask & nogaps_mask):
                tqdm.write(f'skipped {group} on {test_tcr} because it has a gap')
                continue
            elif not nogaps_mask.all():
                ts = tdf[nogaps_mask].tcr.unique().tolist()
                tqdm.write(
                    f'excluding gaps from {group} restricted samples to {len(ts)} TCRs: ' + ', '.join(ts)
                )
                if len(ts) < 10:
                    tqdm.write(f'too few TCRs remain to obtain reliable estimates for {group} on {test_tcr}, skipping.')
                    continue

            # perform successive rounds of shuffling and evaluation
            # since we are performing a leave-tcr-out evaluation, shuffling includes
            # samples from tcrs in the training set (possibly excluding samples with
            # gaps in the position we are shuffling, as per above)
            for i in range(15):
                xshuffle = shuffle(fit_data[nogaps_mask], column_mask)
                shuffle_preds = reg.predict_proba(xshuffle[test_mask[nogaps_mask]])[:, 1]

                pdf = tdf[test_mask & nogaps_mask][[
                    'tcr', 'mut_pos', 'mut_ami', 'wild_activation',
                    'activation', 'is_activated'
                ]]
                pdf['pred'] = shuffle_preds
                pdf['group'] = group
                pdf['shuffle'] = i
                perf.append(pdf)

    ppdf = pd.concat(perf)

    return ppdf


fname = 'results/tcr_stratified_shuffle_importance.csv'
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

#%%

g = sns.catplot(
    data=ddf.query('tcr!="G6" & variable=="auc" & item != "all"'),
    orient='h',
    y='group',
    x='rel',
    col='item',
    sharey=False,
    ci='sd',
    dodge=True,
    aspect=0.7,
    height=5,
    kind='point',
    #hue='tcr',
    zorder=2
)

g.map(sns.stripplot,
    'rel',
    'group',
    'tcr',
    dodge=True,
    hue_order=sorted(ddf['tcr'].unique()),
    palette='husl',
    zorder=1
)

for ax in g.axes_dict.values():
    ax.plot([0, 0], ax.get_ylim(), 'r--')

g.savefig('figures/permutation_feature_importance.pdf', dpi=192)
g.savefig('figures/permutation_feature_importance.png', dpi=192)


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
    order=pddf.groupby(['Position']).agg({'Relative Increase': 'mean'}).sort_values('Relative Increase').index,
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

g.savefig('figures/permutation_feature_importance_positions.pdf', dpi=192)
g.savefig('figures/permutation_feature_importance_positions.png', dpi=192)
