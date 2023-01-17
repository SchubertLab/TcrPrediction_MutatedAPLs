import warnings
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_classification_model(data_tcr, df_tcr, apls_train, idx):
    x_train = data_tcr[df_tcr['epitope'].isin(apls_train)]
    y_train = df_tcr[df_tcr['epitope'].isin(apls_train)]['is_activated']
    clf = RandomForestClassifier(n_estimators=250, max_features='sqrt',
                                 random_state=idx).fit(x_train, y_train)
    return clf


def predict_classification_on_test(data_tcr, df_tcr, clf, apls_train):
    x_test = data_tcr[~df_tcr['epitope'].isin(apls_train)]
    y_test = df_tcr[~df_tcr['epitope'].isin(apls_train)]['is_activated']
    p_test = clf.predict_proba(x_test)
    p_test = p_test[:, (1 if p_test.shape[1] == 2 else 0)]
    return p_test, y_test


def train_regression_model(data_tcr, df_tcr, apls_train, idx):
    x_train = data_tcr[df_tcr['epitope'].isin(apls_train)]
    y_train = df_tcr[df_tcr['epitope'].isin(apls_train)]['activation']
    reg = RandomForestRegressor(n_estimators=250, max_features='sqrt',
                                criterion='mae', random_state=idx).fit(x_train, y_train)
    return reg


def predict_regression_on_test(data_tcr, df_tcr, reg, apls_train):
    x_test = data_tcr[~df_tcr['epitope'].isin(apls_train)]
    y_truth = df_tcr[~df_tcr['epitope'].isin(apls_train)]['activation']
    y_pred = reg.predict(x_test)
    return y_pred, y_truth


def evaluate_classification_models(y_test, p_test, metrics, results, idx):
    # evaluate the classifier for binary metrics
    y_pred = p_test > 0.5
    for name, metric in metrics[1].items():
        try:
            score = metric(y_test, y_pred)
        except ValueError:
            if metric == metrics[1]['auc']:
                score = 1
            else:
                raise ValueError
        results[name].append([idx, score])

    # evaluate the classifier for probability metrics
    for name, metric in metrics[0].items():
        if np.isfinite(p_test).all() and 0 < p_test.mean() < 1:
            try:
                score = metric(y_test, p_test)
            except ValueError:
                if metric == metrics[0]['auc']:
                    score = 1
                else:
                    raise ValueError
        else:
            score = np.nan
        results[name].append([idx, score])
    return


def evaluate_regression_models(y_truth, y_test, metrics, results, idx):
    # evaluate the classifier for probability metrics
    for name, metric in metrics.items():
        score = metric(y_truth, y_test)
        results[name].append([idx, score])
    return


def get_aa_blosum():
    labels = None
    #  Matrix made by matblas from blosum62.iij
    #  * column uses minimum score
    #  BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
    #  Blocks Database = /data/blocks_5.0/blocks.dat
    #  Cluster Percentage: >= 62
    #  Entropy =   0.6979, Expected =  -0.5209
    #  Downloaded from https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt 25/11/2020
    blosum_str = """A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4"""
    blosum_str = blosum_str.replace('  ', ' ')
    matrix = []
    for row in blosum_str.split('\n'):
        if labels is None:
            labels = row.split(' ')
        else:
            mat_row = [int(val) for val in row.split(' ')[1:]]
            matrix.append(mat_row)
    matrix = pd.DataFrame(matrix, columns=labels, index=labels)
    return matrix


def correlation(method):
    def corr_method(y_true, y_pred):
        df = pd.DataFrame()
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        corr = df['y_true'].corr(df['y_pred'], method=method)
        return corr
    return corr_method


def get_metrics_cls():
    mtcs = [
        {
            'auc': metrics.roc_auc_score,
            'aps': metrics.average_precision_score,
        },
        {
            'F1': metrics.f1_score,
            'Accuracy': metrics.accuracy_score,
            'Precision': metrics.precision_score,
            'Recall': metrics.recall_score,
        }
    ]
    return mtcs


def get_metrics_reg():
    metrics_reg = {
        'MAE': metrics.mean_absolute_error,
        'R2': metrics.r2_score,
        'Pearson': correlation(method='pearson'),
        'Spearman': correlation(method='spearman'),
    }
    return metrics_reg
