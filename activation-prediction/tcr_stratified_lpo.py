#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script estimates the feature importance for each position
# using tcr-stratified classification
# the test set contains all apls of a given position for a given tcr
# the train set contains all alps of other positions for other tcrs

import os

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

#%%

df = get_dataset(normalization='AS')

tdf = df[(
    df['mut_pos'] >= 0
) & (
    ~df['cdr3a'].isna()
) & (
    ~df['cdr3b'].isna()
)]
tdf['is_activated'] = tdf['activation'] > 46.9

aa_features = get_aa_features()[['factors']]
#fit_data = full_aa_features(tdf, aa_features, include_tcr=True)


#%% training and evaluation

def train():
    if epitope == 'VPSVWRSSL':
        df = get_tumor_dataset()
    else:
        df = get_dataset(normalization='AS')

    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    )]
    tdf['is_activated'] = tdf['activation'] > 46.9

    aa_features = get_aa_features()[['factors']]
    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1

    perf = []
    for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
        for test_pos in range(8):
            test_mask = (tdf['tcr'] == test_tcr) & (tdf['mut_pos'] == test_pos)
            train_mask = (tdf['tcr'] != test_tcr) & (tdf['mut_pos'] != test_pos)
    
            xtrain = fit_data.loc[train_mask]
            ytrain = tdf.loc[train_mask, 'is_activated']
    
            # train and predict
            reg = RandomForestClassifier(
                n_estimators=250,
                max_features='sqrt',
                n_jobs=n_jobs,
            ).fit(xtrain, ytrain)
            test_preds = reg.predict_proba(fit_data.loc[test_mask])
            
            # save performance
            pdf = tdf[test_mask][[
                'tcr', 'mut_pos', 'mut_ami', 'activation', 'is_activated'
            ]]
            pdf['pred'] = test_preds[:, 1]
            
            perf.append(pdf)
    
        ppdf = pd.concat(perf)

    return ppdf


#%%
epitope = 'VPSVWRSSL'
fname = f'results/{epitope}_tcr_stratified_leave_position_out_performance.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

#%% metrics


pp = pdf.groupby(
    ['tcr', 'mut_pos'], as_index=False
).apply(lambda q: pd.Series({
    'auc': metrics.roc_auc_score(
        q['is_activated'], q['pred']
    ) if 0 < q['is_activated'].mean() < 1 else np.nan,
    'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
}))

pp['is_educated'] = pp['tcr'].str.startswith('ED')

#%%

g = sns.catplot(
    data=pp,
    x='mut_pos',    
    y='auc',
    kind='box',
    #col='is_educated',
    #hue='is_educated',
    palette='husl',
    zorder=2,
    showmeans=True,
    height=3,
    aspect=1.25,
    meanprops={'mfc': 'k', 'mec': 'k'},
)


g.set(
    xticklabels=[f'P{i+1}' for i in range(8)],
    xlabel='Validate on position',
    ylabel='AUC',
)

plt.tight_layout()
plt.savefig(f'figures/{epitope}_tcr_stratified_leave_position_out_performance.pdf', dpi=300,
            bbox_inches='tight')