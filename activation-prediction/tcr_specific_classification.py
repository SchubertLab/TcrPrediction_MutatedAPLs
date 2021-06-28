#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this runs a tcr-specific random forest to classify whether
# a tcr is activated following a certain mutation
# tries all possible combinations of normalization and threshold

import os
import sys
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text  # !pip install adjustText
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm

from preprocessing import (add_activation_thresholds, full_aa_features,
                           get_aa_features, get_complete_dataset)

# %% training

def masked_groupby(df, cols):
    # essentially df.groupby(cols) but returns an iterator of tuples
    # with column values and boolean mask to index the data frame

    for vals in df[cols].drop_duplicates().itertuples(False, None):
        yield vals, reduce(
            lambda a, b: a & b,
            (df[c] == v for c, v in zip(cols, vals))
        )


def tcr_specific_model_classification():
    df = add_activation_thresholds(get_complete_dataset())
    data = df[(
        df['mut_pos'] >= 0
    ) & (
        df['tcr'].isin(df.query('is_activated')['tcr'].unique())
    )]
    aa_features = get_aa_features()

    perf = []
    group_keys = ['normalization', 'threshold', 'tcr']
    for reduce_feats in [False, True]:
        feats = aa_features if reduce_feats else aa_features[['factors']]
        train_data = full_aa_features(data, feats)
        print('training on', train_data.shape[1], 'features')
        
        for _, fit_mask in tqdm(masked_groupby(data, group_keys)):
            fit_data = train_data[fit_mask]
    
            split = KFold(len(fit_data), shuffle=True).split(fit_data)
            for i, (train_idx, test_idx) in enumerate(split):
                xtrain = fit_data.iloc[train_idx]
                xtest = fit_data.iloc[test_idx]
    
                ytrain = data.loc[fit_mask, 'is_activated'].iloc[train_idx]
    
                clf = RandomForestClassifier().fit(xtrain, ytrain)
    
                test_preds = clf.predict_proba(xtest)
                test_preds = test_preds[:, (1 if test_preds.shape[1] == 2 else 0)]
    
                # save performance
                pdf = data[fit_mask].iloc[test_idx][group_keys + [
                    'mut_pos', 'mut_ami', 'is_activated', 'wild_activation'
                ]]
                pdf['fold'] = i
                pdf['pred'] = test_preds
                pdf['reduced_features'] = reduce_feats
                perf.append(pdf)

    # aggregate performance data
    pdf = pd.concat(perf)

    return pdf


fname = 'results/tcr_specific_classification_performance.csv.gz'
if not os.path.exists(fname):
    print('computing results for the first time')
    pdf = tcr_specific_model_classification()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

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

plt.savefig('figures/tcr_specific_feature_comparison.pdf', dpi=192)

# %% separate roc curves

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
        plt.ylabel('TPR / Pr.')
    else:
        plt.yticks([])

    if ax.is_last_row():
        plt.xlabel('FPR / Rec.')
    else:
        plt.xticks([])

plt.gcf().legend(*zip(*[
    (mpl.patches.Patch(facecolor=cm(4 * j + k)), f'{n} / {t}')
    for j, n in enumerate(sorted(pdf['normalization'].unique()))
    for k, t in enumerate(sorted(pdf['threshold'].unique()))
]), title='Normalization / Threshold', loc='lower right', ncol=3)


plt.tight_layout()
sns.despine()
plt.savefig('figures/tcr_specific_activation_aucs_reduced_feats.pdf', dpi=192)


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
plt.savefig('figures/tcr_specific_activation_AS_auroc_auprc_reduced_features.pdf', dpi=192)


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
)
g.map(sns.pointplot, 'threshold', 'value')


def annot(x, y, color, data):
    t = sorted(data[y])[4]
    mask = data[y] <= t
    txt = [
        plt.text(['1', '4', '6'].index(str(thr)[0]), val, tcr)
        for tcr, thr, val in data.loc[mask, ['tcr', 'threshold', 'value']].values
    ]
    adjust_text(txt, arrowprops=dict(arrowstyle='-'))

g.map_dataframe(annot, 'threshold', 'value')

plt.savefig('figures/tcr_specific_thr_vs_norm_reduced_features.pdf', dpi=192)
