#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this trains a random forest validating on a held-out receptor

import os
import sys
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from preprocessing import (add_activation_thresholds, full_aa_features,
                           get_aa_features, get_complete_dataset)

# %% training and evaluation


def masked_groupby(df, cols):
    # essentially df.groupby(cols) but returns an iterator of tuples
    # with column values and boolean mask to index the data frame

    for vals in df[cols].drop_duplicates().itertuples(False, None):
        yield vals, reduce(
            lambda a, b: a & b,
            (df[c] == v for c, v in zip(cols, vals))
        )


def train():
    df = add_activation_thresholds(get_complete_dataset(epitope))

    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    )]

    aa_features = get_aa_features()

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    perf = []
    for reduce_feats in [False, True]:
        feats = aa_features if reduce_feats else aa_features[['factors']]
        fit_data = full_aa_features(tdf, feats, include_tcr=True)
        print('training on', fit_data.shape[1], 'features')
        
        for _, data_mask in tqdm(masked_groupby(tdf, ['normalization', 'threshold'])):
            for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
                train_mask = data_mask & (tdf['tcr'] != test_tcr)
                test_mask = data_mask & (tdf['tcr'] == test_tcr)
    
                if not train_mask.any() or not test_mask.any():
                    continue
    
                xtrain = fit_data[train_mask]
                xtest = fit_data[test_mask]
                ytrain = tdf.loc[train_mask, 'is_activated']
    
                # train and predict
                reg = RandomForestClassifier(
                    n_estimators=1000,
                    n_jobs=n_jobs,
                ).fit(xtrain, ytrain)
                test_preds = reg.predict_proba(xtest)
    
                # save performance
                pdf = tdf[test_mask][[
                    'normalization', 'threshold', 'tcr', 'mut_pos', 'mut_ami',
                    'wild_activation', 'activation', 'is_activated'
                ]]
                pdf['pred'] = test_preds[:, 1]
                pdf['reduced_features'] = reduce_feats
                perf.append(pdf)

    ppdf = pd.concat(perf)

    return ppdf

epitope = 'VPSVWRSSL'

fname = f'results/{epitope}_tcr_stratified_classification_performance.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
    
pdf = pd.read_csv(fname)

# %% load evaluation data
pdf = pdf.query('mut_pos >= 0')
pdf = pdf.groupby([
    'normalization', 'tcr'
]).filter(lambda g: 0 < g['is_activated'].sum() < len(g) - 1)

#%% comparing normalizations

pp = pdf.groupby(
    ['reduced_features', 'normalization', 'tcr', 'threshold'], as_index=False
).apply(lambda q: pd.Series({
    'auc': metrics.roc_auc_score(q['is_activated'], q['pred']),
    'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
})).melt(['reduced_features', 'normalization', 'tcr', 'threshold']).pivot_table(
    'value', ['tcr', 'threshold', 'normalization', 'reduced_features'], 'variable'
).reset_index()

pp.loc[:, 'reduced_features'] = np.where(pp['reduced_features'], 'redux', 'full')

    
#%% are reduced features better?

g = sns.catplot(
    data=pp[(
        (pp['normalization'] == 'AS') & (pp['threshold'] == 46.9)
    ) | (
        (pp['normalization'] == 'OT1') & (pp['threshold'] == 66.8)
    ) | (
        (pp['normalization'] == 'none') & (pp['threshold'] == 15)
    )],
    col='normalization',
    x='reduced_features', y='auc',
    dodge=True, kind='box', margin_titles=True,
    height=3, aspect=0.5,
)

        
def do_test(feats, auc, **kwargs):
    x1 = auc[feats == 'redux']
    x2 = auc[feats == 'full']
    
    r = stats.ttest_rel(x1, x2, alternative='greater')
    
    # bonferroni
    ntests = 3
    if r.pvalue < 0.05 / ntests:
        sig = ' *'
    elif r.pvalue < 0.01 / ntests:
        sig = ' **'
    elif r.pvalue < 0.001 / ntests:
        sig = ' ***'
    else:
        sig = ''
    
    plt.text(0.05, 0.05, f'N = {len(x1)}\nt = {r.statistic:.3f}\np = {r.pvalue:.1e}{sig}',
             backgroundcolor='#ffffff77',
             transform=plt.gca().transAxes, va='bottom')

g.map(do_test, 'reduced_features', 'auc')

plt.savefig(f'figures/{epitope}_tcr_stratified_feature_comparison.pdf', dpi=192)


#%% all aucs together

pdf['is_educated'] = pdf['tcr'].str.startswith('ED')

naive_colors = sns.color_palette(
    'Oranges', n_colors=2 * len(pdf.query('~is_educated')['tcr'].unique()) + 4
)
naive_idx = 1

educated_colors = sns.color_palette(
    'Blues', n_colors=2 * len(pdf.query('is_educated')['tcr'].unique()) + 4
)
educated_idx = 1

plt.figure(figsize=(4 * 1.25, 4))
groups = pdf.query('reduced_features & normalization == "AS" & threshold == 46.9').groupby(['reduced_features', 'tcr'])
for i, ((fs, tcr), g) in enumerate(groups):
    fpr, tpr, _ = metrics.roc_curve(g['is_activated'], g['pred'])
    auc = metrics.roc_auc_score(g['is_activated'], g['pred'])

    if tcr.startswith('ED'):
        educated_idx += 1
        c = educated_colors[educated_idx]
    else:
        naive_idx += 1
        c = naive_colors[naive_idx]

    plt.plot(fpr, tpr, c=c)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

plt.ylabel('TPR')
plt.xlabel('FPR')

plt.legend([
    mpl.lines.Line2D([], [], c='C0'),
    mpl.lines.Line2D([], [], c='C1')
], [
    'Yes', 'No'
], loc='lower right', title='Is Educated')

plt.tight_layout()
sns.despine()
plt.savefig(f'figures/{epitope}_tcr_stratified_activation_AS_auroc_together.pdf', dpi=192)
plt.savefig(f'figures/{epitope}_tcr_stratified_activation_AS_auroc_together.png', dpi=192)

# %% roc auc curves for all thresholds / normalization combinations

ntcrs = len(pdf['tcr'].unique())
ncols = 8
height = 2
nrows = ntcrs // ncols + 1
cm = plt.get_cmap('tab20c')

plt.figure(figsize=(ncols * height, nrows * height))
for i, (tcr, g) in enumerate(pdf.query('reduced_features').groupby('tcr')):
    plt.subplot(nrows, ncols, i + 1)

    best = 0.0
    for j, (norm, h) in enumerate(g.groupby('normalization')):
        for k, (thr, g) in enumerate(h.groupby('threshold')):
            fpr, tpr, _ = metrics.roc_curve(g['is_activated'], g['pred'])
            auc = metrics.roc_auc_score(g['is_activated'], g['pred'])
            best = max(best, auc)
            plt.plot(fpr, tpr, c=cm(4 * j + k), label=f'{norm}/{thr}')

    plt.title(f'{tcr} - Best: {best:.2f}')

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    ax = plt.gca()
    if ax.is_first_col():
        plt.ylabel('TPR')
    else:
        plt.yticks([])

    if ax.is_last_row():
        plt.xlabel('FPR')
    else:
        plt.xticks([])

plt.gcf().legend(*zip(*[
    (mpl.patches.Patch(facecolor=cm(4 * j + k)), f'{n} / {t}')
    for j, n in enumerate(sorted(pdf['normalization'].unique()))
    for k, t in enumerate(sorted(pdf['threshold'].unique()))
]), title='Normalization / Threshold', loc='lower right', ncol=3)

plt.tight_layout()
sns.despine()
plt.savefig(f'figures/{epitope}_tcr_stratified_activation_aucs.pdf', dpi=192)

# %% roc curves for AS / 46.9

ntcrs = len(pdf['tcr'].unique())
ncols = 8
height = 2
nrows = ntcrs // ncols + 1

plt.figure(figsize=(ncols * height, nrows * height))
groups = pdf.query('reduced_features & normalization == "AS" & threshold == 46.9').groupby('tcr')
for i, (tcr, g) in enumerate(groups):
    plt.subplot(nrows, ncols, i + 1)

    fpr, tpr, _ = metrics.roc_curve(g['is_activated'], g['pred'])
    pre, rec, _ = metrics.precision_recall_curve(g['is_activated'], g['pred'])

    auc = metrics.roc_auc_score(g['is_activated'], g['pred'])
    aps = metrics.average_precision_score(g['is_activated'], g['pred'])

    plt.plot(fpr, tpr)
    plt.plot(rec, pre)

    plt.title(f'{tcr}\nAUC: {auc:.2f} - APS: {aps:.2f}')

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    ax = plt.gca()
    if ax.is_first_col():
        plt.ylabel('TPR / Pr.')
    else:
        plt.yticks([])

    if ax.is_last_row():
        plt.xlabel('FPR / Rec.')
    else:
        plt.xticks([])

plt.tight_layout()
sns.despine()
plt.savefig(f'figures/{epitope}_tcr_stratified_activation_AS_auroc_auprc.pdf', dpi=192)


#%% comparing thresholds and normalizations

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


adf = pdf.query('reduced_features').groupby([
    'normalization', 'threshold', 'tcr'
]).apply(
    classification_metrics
).reset_index().melt([
    'normalization', 'threshold', 'tcr'
])

g = sns.catplot(
    data=adf,
    x='threshold', y='value',
    col='normalization',
    row='variable', height=2,
    margin_titles=True,
    order=[15.0, 46.9, 66.8]
)
g.map(sns.pointplot, 'threshold', 'value', order=[15.0, 46.9, 66.8])


def annot(x, y, color, data):
    t = sorted(data[y])[5]
    mask = data[y] <= t
    txt = [
        plt.text(['1', '4', '6'].index(str(thr)[0]), val, tcr)
        for tcr, thr, val in data.loc[mask, ['tcr', 'threshold', 'value']].values
    ]
    adjust_text(txt, arrowprops=dict(arrowstyle='-'))


g.map_dataframe(annot, 'threshold', 'value')

plt.savefig(f'figures/{epitope}_tcr_stratified_thr_vs_norm.pdf', dpi=192)


