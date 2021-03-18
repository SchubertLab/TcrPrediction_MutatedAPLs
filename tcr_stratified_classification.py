#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this trains a random forest validating on a held-out receptor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_dataset, get_aa_features, full_aa_features, get_aa_factors
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
import os
from tqdm import tqdm


def train():
    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    df['is_activated'] = (df['activation'] > 15).astype(np.int64)
    tdf = df.query('mut_pos>=0')
    fit_data = full_aa_features(tdf, aa_features, interactions=False,
                                include_tcr=True)
    print('total features', fit_data.shape[1])

    perf = []
    imps = []
    for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
        test_mask = tdf['tcr'] == test_tcr

        xtrain = fit_data[~test_mask]
        xtest = fit_data[test_mask]
        ytrain = tdf.loc[~test_mask, 'is_activated']

        # train and predict
        reg = RandomForestClassifier(
            n_estimators=1000,
            n_jobs=n_jobs,
        ).fit(xtrain, ytrain)
        test_preds = reg.predict_proba(xtest)

        # save performance
        pdf = tdf[test_mask][[
            'tcr', 'mut_pos', 'mut_ami', 'wild_activation',
            'activation', 'is_activated'
        ]]
        pdf['pred'] = test_preds[:, 1]
        perf.append(pdf)

        # save feature importances
        fidf = pd.DataFrame({
            'imp': reg.feature_importances_,
            'feat': fit_data.columns,
        })
        fidf['tcr'] = test_tcr
        imps.append(fidf)

    ppdf = pd.concat(perf)
    fidf = pd.concat(imps)

    return ppdf, fidf


#%% training and evaluation

if not os.path.exists('classification_performance.csv'):
    df = get_dataset()

    # balancing dataset gives slightly worse performance
    #df = df[df['tcr'].isin({'F4', 'B13', 'B11', 'OT1', 'B15'})]

    aa_features = get_aa_features()
    pdf, fidf = train()

    pdf.to_csv('classification_performance.csv', index=False)
    fidf.to_csv('classification_feat_importance.csv', index=False)
else:
    print('using cached results')

#%% load evaluation data
pdf = pd.read_csv('classification_performance.csv')
pdf['is_activated'] = pdf['activation']>15
pdf = pdf.query('mut_pos >= 0')
pdf = pdf[pdf['tcr'].isin(pdf.query('is_activated')['tcr'].unique())]

#%% predicted probabilities
g = sns.FacetGrid(pdf, col='tcr', col_wrap=2, ylim=(0, 1))
g.map(sns.stripplot,  'is_activated', 'pred', 'mut_pos',
      order=[False, True], hue_order=range(8), palette='husl')
g.map(sns.pointplot, 'is_activated', 'pred', color='C3', order=[False, True])
for ax in g.axes:
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
g.add_legend()
plt.savefig('figures/tcr_activation_prediction.pdf', dpi=192)

#%% roc auc curves

plt.figure(figsize=(6.6, 9))
for i, (tcr, q) in enumerate(pdf.groupby('tcr')):
    plt.subplot(3, 2, i + 1)
    fpr, tpr, thr = roc_curve(q['is_activated'], q['pred'])
    auc = roc_auc_score(q['is_activated'], q['pred'])
    plt.plot(fpr, tpr)
    plt.title(f'{tcr} - {auc:.3f}')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    if i == 0 or i == 3:
        plt.ylabel('TPR')
    else:
        plt.yticks([])
    if i > 2:
        plt.xlabel('FPR')
    else:
        plt.xticks([])

plt.tight_layout()
sns.despine()
plt.savefig('figures/tcr_activation_aucs.pdf', dpi=192)

#%% auprc curves

for i, (tcr, q) in enumerate(pdf.groupby('tcr')):
    plt.subplot(2, 3, i + 1)
    pr, rc, thr = precision_recall_curve(q['is_activated'], q['pred'])
    aps = average_precision_score(q['is_activated'], q['pred'])
    plt.plot(rc, pr)
    plt.title(f'{tcr} - {aps:.3f}')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    if i == 0 or i == 3:
        plt.ylabel('Precision')
    else:
        plt.yticks([])
    if i > 2:
        plt.xlabel('Recall')
    else:
        plt.xticks([])

plt.tight_layout()
sns.despine()
plt.savefig('figures/tcr_activation_auprcs.pdf', dpi=192)

#%% feature importances
fidf = pd.read_csv('classification_feat_importance.csv')
fidf['pos'] = fidf['feat'].str.split('$').str[:1].str.join('$')

ff = fidf.groupby(['tcr', 'feat']).agg({'imp': 'sum'}).reset_index()
ff['pos'] = ff['feat'].str.split('$').str[:1].str.join('$')

print('raw importance')
print(ff.sort_values('imp', ascending=False).head(25))

print('importance by position')
print(
      ff.groupby('pos')
          .agg({'imp': 'sum'})
          .sort_values('imp', ascending=False)
          .head(25)
)


#%% importance plots

q = fidf.groupby(['tcr', 'pos']).agg({'imp': 'sum'}).reset_index()
q['rank'] = q.groupby('tcr').rank(ascending=False)

sns.lineplot(
    x='tcr',
    y='rank',
    hue='pos',
    data=q.query('rank<10').sort_values('rank')
)

sns.lineplot(
    x='tcr',
    y='imp',
    hue='pos',
    data=q.query('rank<10')
)
