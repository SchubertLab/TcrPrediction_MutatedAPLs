#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this runs a tcr-specific random forest to classify whether
# a tcr is activated following a certain mutation

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_dataset, get_aa_features, full_aa_features

#%% training

def tcr_specific_model_classification():
    df = get_dataset()
    df['is_activated'] = df['activation'] > 15
    data = df[(
        df['mut_pos'] >= 0
    ) & (
        df['tcr'].isin(df.query('is_activated')['tcr'].unique())
    )]
    aa_features = get_aa_features()
    train_data = full_aa_features(data, aa_features)
    print('training on', train_data.shape[1], 'features')

    perf = []
    for t in tqdm(data['tcr'].unique()):
        fit_mask = (data['tcr'] == t)
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
            pdf = data[fit_mask].iloc[test_idx][[
                'mut_pos', 'mut_ami', 'is_activated', 'wild_activation'
            ]]
            pdf['tcr'] = t
            pdf['fold'] = i
            pdf['pred'] = test_preds
            perf.append(pdf)

    # aggregate performance data
    pdf = pd.concat(perf)

    return pdf


fname = 'results/tcr_specific_classification_performance.csv'
if not os.path.exists(fname):
    print('computing results for the first time')
    pdf = tcr_specific_model_classification()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

#%% probability plots

g = sns.FacetGrid(pdf, col='tcr', col_wrap=9, ylim=(0, 1), height=2)
g.map(sns.stripplot,  'is_activated', 'pred', 'mut_pos',
      order=[False, True], hue_order=range(8), palette='husl')
g.map(sns.pointplot, 'is_activated', 'pred', color='C3', order=[False, True])
for ax in g.axes:
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
g.add_legend()
plt.savefig('figures/tcr_specific_activation_prediction.pdf', dpi=192)

#%% separate roc curves

ntcrs = len(pdf['tcr'].unique())
ncols = 8
height = 2.5
nrows = ntcrs // ncols + 1

plt.figure(figsize=(ncols * height, nrows * height))
for i, (tcr, q) in enumerate(pdf.groupby('tcr')):
    plt.subplot(nrows, ncols, i + 1)
    fpr, tpr, thr = roc_curve(q['is_activated'], q['pred'])
    auc = roc_auc_score(q['is_activated'], q['pred'])

    pr, rc, thr = precision_recall_curve(q['is_activated'], q['pred'])
    aps = average_precision_score(q['is_activated'], q['pred'])

    plt.plot(fpr, tpr)
    plt.plot(rc, pr)
    plt.title(f'{tcr}\nAUC: {auc:.2f} APS: {aps:.2f}')

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
plt.savefig('figures/tcr_specific_activation_aucs.pdf', dpi=192)

#%% roc curves together

colors = sns.color_palette("husl", len(pdf['tcr'].unique()))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, (tcr, q) in enumerate(pdf.groupby('tcr')):
    fpr, tpr, thr = roc_curve(q['is_activated'], q['pred'])
    plt.plot(fpr, tpr, c=colors[i], label=tcr)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('TPR')
plt.xlabel('FPR')

plt.subplot(1, 2, 2)
for i, (tcr, q) in enumerate(pdf.groupby('tcr')):
    pr, rc, thr = precision_recall_curve(q['is_activated'], q['pred'])
    plt.plot(rc, pr, c=colors[i], label=tcr)

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('Precision')
plt.xlabel('Recall')

plt.tight_layout()
sns.despine()
plt.savefig('figures/tcr_specific_activation_aucs_together.pdf', dpi=192)
plt.savefig('figures/tcr_specific_activation_aucs_together.png', dpi=600)
