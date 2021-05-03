#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this does a random hyperparameter search for gradient boosting on B11

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocessing import full_aa_features, get_aa_features, get_dataset

df = get_dataset()
aa_features = get_aa_features()


def test_gb(gb_kwargs, verbose=False):
    def log(*args):
        if verbose:
            print(*args)

    tdf = df.query('tcr=="B11" & mut_pos>=0')
    fit_data = full_aa_features(tdf, aa_features)
    log('total features', fit_data.shape[1])

    perf = []
    split = KFold(len(fit_data), shuffle=True).split(fit_data)
    for i, (train_idx, test_idx) in enumerate(split):
        xtrain = fit_data.iloc[train_idx]
        xtest = fit_data.iloc[test_idx]
        ytrain = tdf['residual'].iloc[train_idx].values.reshape((-1, 1))

        # standardize data
        xss = StandardScaler().fit(xtrain)
        yss = StandardScaler().fit(ytrain)

        xtrain = xss.transform(xtrain)
        xtest = xss.transform(xtest)
        ytrain = yss.transform(ytrain)

        # reduce dimensionality
        pca = PCA().fit(xtrain)
        var_redux = gb_kwargs.pop('var_redux', 0.995)
        n = np.sum(np.cumsum(pca.explained_variance_ratio_) <= var_redux)
        xtrain = pca.transform(xtrain)[:, :n]
        xtest = pca.transform(xtest)[:, :n]
        log('reduced to {} features with {:.1f}% total variance',
            xtrain.shape[1], 100 * var_redux)

        # train and predict
        clf = GradientBoostingRegressor #if False else RandomForestRegressor
        reg = clf(**gb_kwargs).fit(xtrain, ytrain.reshape(-1))
        test_preds = reg.predict(xtest)
        test_preds = yss.inverse_transform(test_preds.reshape((-1, 1)))

        # save performance
        pdf = tdf.iloc[test_idx][[
            'tcr', 'mut_pos', 'mut_ami', 'wild_activation',
            'activation', 'residual'
        ]]
        pdf['pred_res'] = test_preds
        perf.append(pdf)
        log(pdf)

    ppdf = pd.concat(perf)
    ppdf['pred'] = ppdf['pred_res'] + ppdf['wild_activation']
    ppdf['err'] = ppdf['pred_res'] - ppdf['residual']

    return ppdf, gb_kwargs

#%% run random search

best = None
kws = [{
    'n_estimators': np.random.choice(range(100, 1001)),
    'max_depth': np.random.choice(range(2, 11)),
    'learning_rate': 10**(np.random.random() * 4 - 4),
    'min_samples_split': np.random.choice(range(2, 25)),
    'var_redux': np.random.choice([0.5, 0.75, 0.9, 0.95, 0.99, 0.999]),
} for i in range(100)]

it = (test_gb(k) for k in kws)
for i, (ppdf, gb_kws) in enumerate(it):
    mae = np.mean(np.abs(ppdf['err']))
    if best is None or mae < best[0]:
        best = mae, gb_kws

    print(
        f'Done {i} iterations, last MAE {mae:.4f}, best MAE {best[0]:.4f}'
    )

print(best)
