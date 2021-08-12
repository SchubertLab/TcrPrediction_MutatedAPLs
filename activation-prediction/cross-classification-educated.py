#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script finds the predictive performance on naive tcrs when
# training on educated tcrs

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

from preprocessing import (full_aa_features, get_aa_factors, get_aa_features,
                           get_dataset)

#%% training and evaluation
def train():
    df = get_dataset(normalization='AS')
    df['is_activated'] = df['activation'] > 46.9
    
    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    ) & (
        df['tcr'].isin(df.query('is_activated')['tcr'].unique())
    )]
    
    aa_features = get_aa_features()
    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)
    print('total features', fit_data.shape[1])
    
    tdf['is_educated'] = tdf['tcr'].str.startswith('ED')
    
    train_mask = tdf['is_educated']
    test_mask = ~tdf['is_educated']
    
    reg = RandomForestClassifier(
        n_estimators=1000,
    ).fit(fit_data[train_mask], tdf.loc[train_mask, 'is_activated'])
    test_preds = reg.predict_proba(fit_data)
    
    # save performance
    pdf = tdf[[
        'mut_pos', 'mut_ami', 'wild_activation',
        'activation', 'is_activated', 'tcr',
        'is_educated'
    ]]
    pdf['pred'] = test_preds[:, 1]

    return pdf


fname = 'results/cross-performance-educated-vs-naive.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)


#%%


pp = pdf.query('~is_educated').groupby(
    'tcr', as_index=False
).apply(lambda q: pd.Series({
    'auc': roc_auc_score(q['is_activated'], q['pred']),
    'aps': average_precision_score(q['is_activated'], q['pred']),
}))

# g = sns.catplot(
#     data=pp.sort_values('auc', ascending=False), y='tcr', x='auc',
#     aspect=0.8, height=2.5, orient='h'
# )

g = sns.catplot(
    data=pp.sort_values('auc', ascending=False), x='tcr', y='auc',
    aspect=2, height=1.5
)


g.ax.tick_params(axis='x', rotation=90)    

g.savefig('figures/cross-classification-educated.pdf', dpi=300)