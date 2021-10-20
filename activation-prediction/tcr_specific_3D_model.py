# this runs a tcr-specific random forest to classify whether
# a tcr is activated following a certain mutation
# based on the distance matrix of TCRpMHCmodels

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


def get_distances():
    data = pd.read_csv('../Modelling3D/Output/features/distances_B11.csv', index_col=0)
    data = data.sort_index()
    data = data[data.index != 'SIINFEKL']

    return data


def get_labels():
    activations = pd.read_csv('../data/activations_lena.csv', index_col=0)
    activations = activations.sort_index()
    activations['is_activated'] = activations['B11'] > 15
    activations = activations[activations.index != 'SIINFEKL']
    return activations


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

    train_data = train_data[data['tcr'] == 'B11']
    data = data[data['tcr'] == 'B11']
    train_data['epitope'] = data['epitope']
    train_data = train_data.set_index('epitope')
    train_data = train_data.sort_index()
    train_data.index.name = None
    # for el in data.columns:
    #     print(el)

    fit_data = get_distances()
    fit_label = get_labels()

    fit_data[list(train_data.columns)] = train_data.values

    predictions = []
    ground_truth = []

    split = KFold(len(fit_data), shuffle=True).split(fit_data)
    for i, (train_idx, test_idx) in enumerate(tqdm(split)):
        xtrain = fit_data.iloc[train_idx]
        xtest = fit_data.iloc[test_idx]

        ytrain = fit_label.iloc[train_idx]['is_activated']
        ytest = fit_label.iloc[test_idx]['is_activated']

        clf = RandomForestClassifier().fit(xtrain, ytrain)

        test_preds = clf.predict_proba(xtest)
        test_preds = test_preds[:, (1 if test_preds.shape[1] == 2 else 0)]

        predictions.append(test_preds)
        ground_truth.append(ytest)

    return predictions, ground_truth


preds, labels = tcr_specific_model_classification()

# roc curves
plt.figure(figsize=(8, 8))

fpr, tpr, _ = roc_curve(labels, preds)
auc = roc_auc_score(labels, preds)

pr, rc, _ = precision_recall_curve(labels, preds)
aps = average_precision_score(labels, preds)

plt.plot(fpr, tpr)
plt.plot(rc, pr)
plt.title(f'B11\nAUC: {auc:.2f} APS: {aps:.2f}')

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
plt.savefig('figures/tcr_specific_activation_aucs_spatial.pdf', dpi=600)
