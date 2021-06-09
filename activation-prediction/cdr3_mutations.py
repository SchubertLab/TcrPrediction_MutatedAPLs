#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script will find which mutations in the CDR3 will cause the
# largest change in activations

import os

import concurrent.futures
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
                           get_complete_dataset, get_dataset)

#%% training and prediction

def predict_for_mutation(predictor, test_df, aa_features, chain, pos, amino_from, amino_to):
    # perform mutation
    seq = test_df[f'cdr3{chain }_aligned'].iloc[0]
    mutated = list(seq)
    mutated[pos] = amino_to
    mdf = test_df.copy()
    mdf.loc[:, f'cdr3{chain}_aligned'] = ''.join(mutated)
    
    # predict
    feats = full_aa_features(mdf, aa_features, include_tcr=True)
    mdf.loc[:, 'pred'] = predictor.predict(feats)
    
    # save
    mdf.loc[:, 'cdr3_mutation_chain'] = chain
    mdf.loc[:, 'cdr3_mutation_position'] = pos
    mdf.loc[:, 'cdr3_mutation_amino'] = amino_to
    mdf.loc[:, 'cdr3_mut'] = f'{chain}{amino_from}{pos}{amino_to}'

    return mdf


def wrapper(args):
    return predict_for_mutation(*args)


def train():
    df = get_dataset(normalization='AS', cdr3_alignment_type='imgt')
    
    tdf = df[(
        df['mut_pos'] >= 0
    ) & (
        ~df['cdr3a'].isna()
    ) & (
        ~df['cdr3b'].isna()
    )]
        
    aa_features = get_aa_features()
    fit_data = full_aa_features(tdf, aa_features, include_tcr=True)

    # disable parallel processing if running from within spyder
    n_jobs = 1 if 'SPY_PYTHONPATH' in os.environ else -1
    
    perf = []
    for test_tcr in tqdm(tdf['tcr'].unique(), ncols=50):
        test_mask = tdf['tcr'] == test_tcr
    
        xtrain = fit_data.loc[~test_mask]
        xtest = fit_data.loc[test_mask]
        ytrain = tdf.loc[~test_mask, 'activation']
    
        # train and predict on test data
        reg = RandomForestRegressor(
            n_jobs=n_jobs,
            n_estimators=250,
            max_features='sqrt',
            criterion='mae',
        ).fit(xtrain, ytrain)
        test_preds = reg.predict(xtest)

        # save performance
        pdf = tdf[test_mask][[
            'normalization', 'tcr', 'mut_pos', 'mut_ami', 'activation',
        ]]
        pdf['pred'] = test_preds
        perf.append(pdf)
        
        task_list = [
            (reg, tdf[test_mask], aa_features, chain, i, a, b)
            for chain in ['a', 'b']
            for i, a in enumerate(tdf[test_mask][f'cdr3{chain}_aligned'].iloc[0])
            if a != '-'
            for b in 'ARNDCQEGHILKMFPSTWYV'
            if b != a
        ]

        if n_jobs == 1:
            for args in tqdm(task_list, ncols=50, leave=False):
                mdf = wrapper(args)
                perf.append(mdf)
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(wrapper, task_list)
                for mdf in tqdm(results, total=len(task_list), ncols=50, leave=False):
                    perf.append(mdf)

    return pd.concat(perf)


#%%
fname = 'results/cdr3_mutations.csv.gz'
if not os.path.exists(fname):
    pdf = train()
    pdf.to_csv(fname, index=False)
else:
    print('using cached results')
    pdf = pd.read_csv(fname)


#%% preprocess data

edf = pd.merge(
    pdf[~pdf['cdr3_mut'].isna()],
    pdf[pdf['cdr3_mut'].isna()][[
        'normalization', 'tcr', 'mut_pos', 'mut_ami', 'pred',
    ]],
    on=['normalization', 'tcr', 'mut_pos', 'mut_ami'],
    suffixes=('_base', '_mut'),
)

edf['diff'] = edf['pred_mut'] - edf['pred_base']
edf['absdiff'] = np.abs(edf['diff'])
edf['epi_mut'] = edf['mut_pos'].astype(str).str.cat(edf['mut_ami'])
edf['is_educated'] = edf['tcr'].str.startswith('ED')
edf['cdr3_mutation_position'] = edf['cdr3_mutation_position'].astype(int)

#%% find largest change mutations

def summ(g, n):
    g = g.sort_values('diff')
    return pd.concat([g.head(n), g.tail(n)])[[
        'epi_mut', 'cdr3_mut', 'pred_base', 'pred_mut', 'diff',
    ]]


print(
      edf.groupby('tcr').apply(lambda g: summ(g, 1)).reset_index().drop(columns='level_1')
)

#%% line plot

sns.catplot(
    data=edf,
    x='cdr3_mutation_position', y='diff', col='cdr3_mutation_chain',
    hue='tcr', kind='point', ci=None, legend=False, markers='None',
    color='navy', alpha=0.1,
)


#%% boxplot

sns.catplot(
    data=edf,
    x='cdr3_mutation_position', y='absdiff',
    col='cdr3_mutation_chain', col_wrap=1,
    kind='box', height=2.5, aspect=2,
)

plt.savefig('figures/cdr3_all_positions.pdf')

#%% 

tdf = edf.query(
    #'tcr=="ED5" & cdr3_mutation_chain == "a" & cdr3_mutation_position == 12'
    #'cdr3_mutation_chain == "a" & cdr3_mutation_position == 12'
    'cdr3_mutation_chain == "a"'    
).sort_values(['cdr3_mut', 'mut_pos', 'mut_ami'])

g = sns.catplot(data=tdf,
                x='mut_pos', y='diff', hue='tcr',
                dodge=True, kind='point', ci=None,
                col='cdr3_mutation_position',
                col_order=tdf.groupby('cdr3_mutation_position')['diff'].agg('std').sort_values().index.tolist(),
                col_wrap=4, height=2.5,
                #col='tcr', col_wrap=8,
                hue_order=tdf.groupby('tcr')['diff'].agg('mean').sort_values().index.tolist(),
                )


#%% two positions with largest variation for each chain
edf['cdr3_mut_loc'] = edf['cdr3_mutation_chain'].str.cat(edf['cdr3_mutation_position'].astype(str))
locs = edf.groupby('cdr3_mut_loc')['diff'].agg('std').sort_values().tail(4).index.tolist()


g = sns.catplot(
    data=edf[edf['cdr3_mut_loc'].isin(locs)],
    x='mut_pos', y='diff', hue='tcr',
    dodge=False, kind='point', ci=None, marker='None',
    #dodge=True, kind='strip', # cool but slow
    col='cdr3_mut_loc',
    col_wrap=2, height=3,
    hue_order=tdf.groupby('tcr')['diff'].agg('mean').sort_values().index.tolist(),
    legend=False
)

g.add_legend(ncol=2)
g.savefig('figures/cdr3_interesting_positions.pdf')


