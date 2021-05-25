#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script predicts the effect of k-mutations (k>1) using
# tcr-specific models trained with k=1 mutation data

import os
import sys
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text  # !pip install adjustText
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools

from preprocessing import (add_activation_thresholds, full_aa_features,
                           get_aa_features, get_complete_dataset, get_dataset)


# %% training
def get_k_mutations(k, epitope='SIINFEKL', aminos='ARNDCQEGHILKMFPSTWY'):
    if k <= 0:
        yield epitope
    else:
        for aminos in itertools.product(list(aminos), repeat=k):
            for positions in itertools.combinations(range(len(epitope)), k):
                mut = list(epitope)
                for a, i in zip(aminos, positions):
                    if epitope[i] != a:
                        mut[i] = a
                    else:
                        # this choice of positions and amino acids would not
                        # result in k mutations, therefore skip it
                        break
                else:
                    # yield only if we made exactly k changes
                    yield ''.join(mut)


def k_mutation_prediction(k=2):
    df = pd.concat([
        get_dataset(educated_repertoire=True, normalization='AS'),
        get_dataset(educated_repertoire=False, normalization='AS'),
    ])

    df['is_activated'] = df['activation'] > 46.9
    df = df.drop(columns=[
        'activation', 'wild_activation', 'residual',
        'mut_pos', 'mut_ami', 'orig_ami', 'normalization',
        'cdr3a', 'cdr3b', 'cdr3a_aligned', 'cdr3b_aligned',
    ])
    data = df[df['tcr'].isin(df.query('is_activated')['tcr'].unique())]

    aa_features = get_aa_features()
    train_data = full_aa_features(data, aa_features, include_mutation=False,
                                  remove_constant=False)
    print('training on', train_data.shape[1], 'features')

    perf = []
    for tcr in tqdm(data['tcr'].unique()):
        train_mask = data['tcr'] == tcr

        # train
        xtrain = train_data[train_mask]
        ytrain = data.loc[train_mask, 'is_activated']
        clf = RandomForestClassifier().fit(xtrain, ytrain)

        # predict
        mut_data = pd.DataFrame([{
            'tcr': tcr, 'epitope': mut
        } for mut in get_k_mutations(k)])
        mut_feats = full_aa_features(mut_data, aa_features, include_mutation=False,
                                     remove_constant=False)
        mut_data['pred'] = clf.predict_proba(mut_feats)[:, 1]
        mut_data['mutations'] = 2

        # predict for training too and save
        orig_data = data[train_mask]
        orig_data['pred'] = clf.predict_proba(xtrain)[:, 1]
        orig_data['mutations'] = 1

        perf.append(mut_data)
        perf.append(orig_data)

    pdf = pd.concat(perf)

    return pdf


fname = 'results/k_mutation_predictions.csv.gz'
if not os.path.exists(fname):
    print('computing results for the first time')
    pdf = k_mutation_prediction(k=2)
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)

#%% convert epitope to APL

pdf.loc[pdf['epitope'] == 'SIINFEKL', 'mutations'] = 0

mut_to_apl = {
    (p, a): i + 1
    for i, (p, a) in enumerate([
        (p, a)
        for p in range(8)
        for a in 'ACDEFGHIKLMNPQRSTVWY'
        if a != 'SIINFEKL'[p]
    ])
}

mdf = []
for epi in pdf['epitope'].unique():
    m1 = m2 = -1, ''
    for i, (a, b) in enumerate(zip(epi, 'SIINFEKL')):
        if a != b:
            if m1[0] < 0:
                m1 = i, a
            else:
                m2 = i, a

    mdf.append({
        'epitope': epi,
        'mut1_pos': m1[0],
        'mut1_ami': m1[1],
        'mut1_apl': mut_to_apl.get(m1),
        'mut2_pos': m2[0],
        'mut2_ami': m2[1],
        'mut2_apl': mut_to_apl.get(m2),
    })

mdf = pd.DataFrame(mdf)
pdf = pdf.merge(mdf)

#%% raw plot for ED5

def make_plot(g):
    for _, row in g.iterrows():
        plt.plot(
            [
                'ACDEFGHIKLMNPQRSTVWY'.index(row[f'mut1_ami']),
                'ACDEFGHIKLMNPQRSTVWY'.index(row[f'mut2_ami']),
            ],
            [0, 1],
            c=cmap(row['pred']),
            alpha=0.4,
        )

    p1 = g.iloc[0]['mut1_pos']
    p2 = g.iloc[0]['mut2_pos']
    plt.title(f'{p1} - {p2}')
    plt.xticks([])
    plt.yticks([])


plt.figure(figsize=(8 * 2, 8 * 2))
cmap = plt.get_cmap('inferno')
for i, ((p1, p2), g) in enumerate(pdf.query('tcr == "ED5" & mutations == 2').groupby(['mut1_pos', 'mut2_pos'])):
    plt.subplot(8, 8, 8 * p1 + p2 + 1)
    make_plot(g)

    plt.subplot(8, 8, 8 * p2 + p1 + 1)
    make_plot(g)


for p, g in pdf.query('tcr == "ED5" & mutations == 1').groupby('mut1_pos'):
    plt.subplot(8, 8, 1 + 9 * p)
    vals = g.sort_values('mut1_ami')['pred'].values
    plt.bar(range(len(g)), vals,
            color=[cmap(v) for v in vals])
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(p))

plt.tight_layout()

#%% check to which degree predictions use logical operator

pdf['pred_act'] = pdf['pred'] >  0.5
apl1 = pdf[(
    ~pdf['mut1_apl'].isna()
) & (
    pdf['mutations'] == 1
)][['tcr', 'mut1_apl', 'pred', 'pred_act']]

apl2 = pdf.query('mutations == 2')[['tcr', 'mut1_apl', 'mut2_apl', 'pred', 'pred_act']]

# symmetrize APL predictions (we only predicted APL1-APL2 but not APL2-APL1,
# but for the purpose of this experiment we want to count them both)
apl2 = pd.concat([
    apl2,
    apl2.copy().rename(columns={
        'mut1_apl': 'mut2_apl',
        'mut2_apl': 'mut1_apl'
    })
])

a12df = apl1.merge(
    # cross-product to generate all possible APLs
    apl1, on='tcr'
).rename(columns={
    'mut1_apl_x': 'mut1_apl',
    'pred_act_x': 'apl1_pred_act',
    'pred_x': 'apl1_pred',
    'mut1_apl_y': 'mut2_apl',
    'pred_act_y': 'apl2_pred_act',
    'pred_y': 'apl2_pred',
}).merge(
    # compare predictions for individual APLs with coupled prediction
    apl2, on=['tcr', 'mut1_apl', 'mut2_apl']
)
    
xdf = a12df.groupby([
    'tcr', 'apl1_pred_act', 'apl2_pred_act'
])[['apl1_pred', 'apl2_pred', 'pred']].agg('mean').reset_index()

# is it an and ?
xdf.loc[xdf['apl1_pred_act'] & xdf['apl2_pred_act'], 'eand'] = 1
xdf.loc[~xdf['apl1_pred_act'] | ~xdf['apl2_pred_act'], 'eand'] = 0
xdf['eand'] -= xdf['pred']
xdf['eand'] *= xdf['eand']

# is it an or ?
xdf.loc[xdf['apl1_pred_act'] | xdf['apl2_pred_act'], 'eor'] = 1
xdf.loc[~xdf['apl1_pred_act'] & ~xdf['apl2_pred_act'], 'eor'] = 0
xdf['eor'] -= xdf['pred']
xdf['eor'] *= xdf['eor']


g = sns.catplot(
    data=xdf.groupby('tcr').agg({'eand': 'mean', 'eor': 'mean'}).sort_values('eand').reset_index().melt('tcr'),
    y='tcr', x='value', aspect=0.6, col='variable', height=4,
)

g.fig.tight_layout()
plt.savefig('figures/2mutation-logical.png', dpi=192)

print(
    xdf.query(
        'tcr=="ED10" | tcr == "B13" | tcr == "ED40"'
    )[[
        'tcr', 'apl1_pred_act', 'apl2_pred_act', 'pred'
    ]].rename(columns={
        'apl1_pred_act': '1st 1-APL',
        'apl2_pred_act': '2nd 1-APL',
        'pred': '2-APL Prob.'
    })
)