#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this runs a default random forest model on B11, B15 and OT1
#


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_dataset, get_aa_features, get_features


df = get_dataset()
df['is_activated'] = df['activation'] > 15
aa_features = get_aa_features()

#%%

def tcr_specific_model_classification(
    use_orig_props: bool,
    use_mutated_props: bool,
    use_diff_props: bool,
    use_sequence_1hot: bool,
    use_sequence_props: bool,
    use_sequence_diff: bool,
    experiment_name: str,
):
    perf = []

    # build training features
    data = df[df['mut_pos'] >= 0]
    train_data = get_features(data, aa_features, use_orig_props,
                              use_mutated_props, use_diff_props,
                              use_sequence_1hot, use_sequence_props,
                              use_sequence_diff)
    print('training on', train_data.shape[1], 'features')

    for t in ['B11', 'B15', 'OT1', 'F4', 'B13', 'G6']:
        fit_mask = (data['tcr'] == t)
        fit_data = train_data[fit_mask]

        split = KFold(len(fit_data), shuffle=True).split(fit_data)
        for i, (train_idx, test_idx) in tqdm(enumerate(split), total=len(fit_data)):
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

    pdf['use_mutated_props'] = use_mutated_props
    pdf['use_sequence_1hot'] = use_sequence_1hot
    pdf['use_sequence_props'] = use_sequence_props

    if experiment_name:
        pdf['features'] = experiment_name

    return pdf

#%% training

pdf = tcr_specific_model_classification(
    use_orig_props=True,
    use_mutated_props=True,
    use_diff_props=True,
    use_sequence_1hot=True,
    use_sequence_props=True,
    use_sequence_diff=True,
    experiment_name='full',
)

pdf.to_csv('tcr_specific_classification_performance.csv', index=False)

#%% load evaluatio data

pdf = pd.read_csv('tcr_specific_classification_performance.csv')


#%%

g = sns.FacetGrid(pdf, col='tcr', col_wrap=2, ylim=(0, 1))
g.map(sns.stripplot,  'is_activated', 'pred', 'mut_pos',
      order=[False, True], hue_order=range(8), palette='husl')
g.map(sns.pointplot, 'is_activated', 'pred', color='C3', order=[False, True])
for ax in g.axes:
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
g.add_legend()
plt.savefig('figures/tcr_specific_activation_prediction.pdf', dpi=192)


#%%

plt.figure(figsize=(6, 9))
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
plt.savefig('figures/tcr_specific_activation_aucs.pdf', dpi=192)
