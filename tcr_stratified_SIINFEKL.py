# this runs a tcr-specific random forest to classify whether
# a tcr is activated following a certain mutation
# based on the distance matrix of TCRpMHCmodels
import ast
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import utils_distance_matrix as Utils


ENCODING_METHOD = 'muscle'
EPITOPE = 'SIIGFEKL'


def tcr_specific_model_classification():

    fit_data = Utils.get_distances(epitope=EPITOPE)
    fit_label = Utils.get_labels(fit_data.index, epitope=EPITOPE, do_binary=True)

    print(sum(fit_label['is_activated'].values))
    print(len(fit_label))

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
plt.title(f'{EPITOPE}_{ENCODING_METHOD}\nAUC: {auc:.2f} APS: {aps:.2f}')

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
plt.savefig(f'figures/tcr_stratified_{EPITOPE}_{ENCODING_METHOD}_aucs.pdf', dpi=600)
