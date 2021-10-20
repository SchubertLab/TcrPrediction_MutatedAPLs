import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from tqdm import tqdm
from itertools import product
import multiprocessing
import time

import json
import os
import sys
import argparse

sys.path.append('..')

from preprocessing import add_activation_thresholds, full_aa_features, get_aa_features
from preprocessing import get_complete_dataset

from utils_al import get_aa_blosum, get_metrics_cls, get_metrics_reg
from utils_al import predict_regression_on_test, evaluate_classification_models, evaluate_regression_models


def train_classification_model(samples_train, seed):
    mask = data['full_sample'].isin(samples_train)
    data_train = data[mask]
    feat_train = features[mask]
    y_train = data_train['is_activated']
    clf = RandomForestClassifier(n_estimators=1000, random_state=seed).fit(feat_train, y_train)
    return clf


def predict_classification_on_test(clf, samples_train):
    mask = ~data['full_sample'].isin(samples_train)
    data_test = data[mask]
    feat_test = features[mask]

    y_test = data_test['is_activated']
    p_test = clf.predict_proba(feat_test)
    p_test = p_test[:, (1 if p_test.shape[1] == 2 else 0)]
    return p_test, y_test


def train_regression_model(samples_train, seed):
    mask = data['full_sample'].isin(samples_train)
    data_train = data[mask]
    feat_train = features[mask]
    y_train = data_train['activation']
    reg = RandomForestRegressor(n_estimators=250, max_features='sqrt', criterion='mae', random_state=seed).fit(feat_train, y_train)
    return reg


def predict_regression_on_test(reg, samples_train):
    mask = ~data['full_sample'].isin(samples_train)
    data_test = data[mask]
    feat_test = features[mask]

    y_test = data_test['activation']
    y_pred = reg.predict(feat_test)
    return y_pred, y_test


def add_greedy_across(samples_train, clf, reg, N):
    max_score = 0.
    best_tcr = ''

    samples_test = data[~data['full_sample'].isin(samples_train)]['full_sample']

    def score_by_tcr(sample):
        train_new = samples_train.copy()
        train_new.append(sample)

        clf = train_classification_model(samples_train, start_idx)
        p_test, c_truth = predict_classification_on_test(clf, train_new)
        score = metrics.roc_auc_score(c_truth, p_test)
        return score
    scores = {sample: score_by_tcr(sample) for sample in tqdm(samples_test)}
    print(scores)

    new_samples = []
    for idx in range(N):
        best_sample = max(scores, key=scores.get)
        new_samples.append(best_sample)
        print(best_sample)
        print(new_samples)
        scores.pop(best_sample, None)
    print(new_samples)
    return new_samples


def run_learning_loop(tcr, method_data_aquisition,
                      N=8, M=5, seed=0):
    samples_train = data[data['tcr']!=tcr]['full_sample'].values.tolist()

    results = {}
    for metric_name in list(metrics_class[0].keys()) + list(metrics_class[1].keys()) + list(metrics_reg.keys()):
        results[metric_name] = []

    clf = train_classification_model(samples_train, seed)
    reg = train_regression_model(samples_train, seed)

    for idx in range(M):
        samples_train += method_data_aquisition(samples_train, clf, reg, N)

        clf = train_classification_model(samples_train, seed)
        p_test, c_truth = predict_classification_on_test(clf, samples_train)
        evaluate_classification_models(c_truth, p_test, metrics_class, results, idx)

        reg = train_regression_model(samples_train, seed)
        y_test, y_truth = predict_regression_on_test(reg, samples_train)
        evaluate_regression_models(y_truth, y_test, metrics_reg, results, idx)
    return results


def run_experiment(method_data_aquisition, N, M):
    results = {}
    for metric_name in list(metrics_reg.keys()) + list(metrics_class[0].keys()) + list(metrics_class[1].keys()):
        results[metric_name] = []

    for i in tqdm(range(1)):
        for tcr in data['tcr'].unique():
            scores = run_learning_loop(tcr, method_data_aquisition, N=N, M=M, seed=start_idx)
            for name, score in scores.items():
                results[name] += [[tcr] + val for val in score]
            break
    return results


parser = argparse.ArgumentParser()
parser.add_argument('--normalization', type=str, default='AS')
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--threshold', type=str, default='46.9')
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--n_exp', type=int, default=10)
args = parser.parse_args()


NORMALIZATION = args.normalization
THRESHOLD = args.threshold
N = args.n
M = args.m

n_exp = args.n_exp
start_idx = args.start_idx * n_exp

BASE_EPITOPE = 'SIINFEKL'

data = get_complete_dataset()
data = add_activation_thresholds(data)
data['full_sample'] = data[['epitope', 'tcr']].apply(tuple, axis=1)

data = data[data['normalization'] == NORMALIZATION]
data = data[data['threshold'] == THRESHOLD]
data = data[data['is_educated'] == True]

aa_features = get_aa_features()
features_no_mutation = aa_features.loc[['-']]
features_no_mutation.rename(index={'-': None}, inplace=True)
aa_features = pd.concat([aa_features, features_no_mutation])
features = full_aa_features(data, aa_features[['factors']])

metrics_reg = get_metrics_reg()
metrics_class = get_metrics_cls()

resis = run_experiment(add_greedy_across, N, M)
print(resis)

path_base = os.path.dirname(os.path.abspath(__file__))
path_out = path_base + f'/results/al/CROSS_greedy_baseline_{N}_start_{start_idx}_n_{n_exp}.json'

with open(path_out, 'w') as output_file:
    json.dump(resis, output_file)
