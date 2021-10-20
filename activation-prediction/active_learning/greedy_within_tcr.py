import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm import tqdm
import argparse
import json
import os

from preprocessing import add_activation_thresholds, full_aa_features, get_aa_features
from preprocessing import get_complete_dataset
from utils_al import train_classification_model, predict_classification_on_test, train_regression_model
from utils_al import predict_regression_on_test, evaluate_classification_models, evaluate_regression_models

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


def add_gready_upper_bound(data_tcr, df_tcr, apls_train, clf, reg, N):
    new_apls = []
    for idx in range(N):
        max_score = 0.
        best_tcr = ''

        apls_test = df_tcr[~df_tcr['epitope'].isin(apls_train)]['epitope']
        def score_by_tcr(tcr):
            train_new = apls_train.copy()
            train_new = train_new + new_apls
            train_new.append(tcr)

            clf = train_classification_model(data_tcr, df_tcr, train_new, idx)
            p_test, c_truth = predict_classification_on_test(data_tcr, df_tcr, clf, train_new)
            score = metrics.roc_auc_score(c_truth, p_test)
            return score

        scores = {tcr: score_by_tcr(tcr) for tcr in apls_test}
        best_tcr = max(scores, key=scores.get)
        new_apls.append(best_tcr)
    return new_apls


def run_gready_upper_bound(data_tcr, df_tcr, metrics_reg, metrics_class,
                           N=8, M=5, base_epitope='SIINFEKL', seed=0):
    all_apls = df_tcr['epitope'].unique()
    apls_train = add_gready_upper_bound(data_tcr, df_tcr, [base_epitope], None, None, 9)

    results = {}
    for metric_name in list(metrics_class[0].keys()) + list(metrics_class[1].keys()) + list(metrics_reg.keys()):
        results[metric_name] = []

    for idx in range(M):
        clf = train_classification_model(data_tcr, df_tcr, apls_train, seed)
        p_test, c_truth = predict_classification_on_test(data_tcr, df_tcr, clf, apls_train)
        evaluate_classification_models(c_truth, p_test, metrics_class, results, idx)  # todo

        reg = train_regression_model(data_tcr, df_tcr, apls_train, seed)
        y_test, y_truth = predict_regression_on_test(data_tcr, df_tcr, reg, apls_train)
        evaluate_regression_models(y_truth, y_test, metrics_reg, results, idx)

        if (idx + 1) * N + 1 > len(base_epitope) * 19 + 1:
            print('Not enough data left for next step')
            break
        apls_train += add_gready_upper_bound(data_tcr, df_tcr, apls_train, None, None, N)
    return results


def run_experiment(metrics_reg, metrics_class, N, M):
    results = {}
    for metric_name in list(metrics_reg.keys()) + list(metrics_class[0].keys()) + list(metrics_class[1].keys()):
        results[metric_name] = []

    for i in tqdm(range(n_exp)):
        for tcr in tqdm(data['tcr'].unique()):
            mask_tcr = data['tcr'] == tcr
            sequence_rep_tcr =  sequence_representation[mask_tcr]
            data_tcr = data[mask_tcr]

            if sum(data_tcr['is_activated']) == 0 or sum(data_tcr['is_activated'] == len(data_tcr)):
                continue

            scores = run_gready_upper_bound(sequence_rep_tcr, data_tcr,
                                            metrics_reg, metrics_class,
                                            N=N, M=M, seed=i+start_idx)

            tcr_info = [tcr, bool(data_tcr['is_educated'].iloc[0])]
            for name, score in scores.items():
                results[name] += [tcr_info + val for val in score]
    return results


data = get_complete_dataset()
data = add_activation_thresholds(data)

data = data[data['normalization'] == NORMALIZATION]
data = data[data['threshold'] == THRESHOLD]
data = data[data['is_educated'] == True]


aa_features = get_aa_features()
features_no_mutation = aa_features.loc[['-']]
features_no_mutation.rename(index={'-': None}, inplace=True)
aa_features = pd.concat([aa_features, features_no_mutation])

sequence_representation = full_aa_features(data, aa_features[['factors']])

def correlation(method):
    def corr_method(y_true, y_pred):
        df = pd.DataFrame()
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        corr = df['y_true'].corr(df['y_pred'], method=method)
        return corr
    return corr_method

metrics_cls = [
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

metrics_reg = {
    'MAE': metrics.mean_absolute_error,
    'R2': metrics.r2_score,
    'Pearson': correlation(method='pearson'),
    'Spearman': correlation(method='spearman'),
}

resis = run_experiment(metrics_reg, metrics_cls, N, M)
print(resis)

path_base = os.path.dirname(os.path.abspath(__file__))
path_out = path_base + f'/results/greedy_baseline_{N}_start_{start_idx}_n_{n_exp}.json'

with open(path_out, 'w') as output_file:
    json.dump(resis, output_file)

