import Modelling3D.PositionalDistances as Dist

import os

import pandas as pd
import numpy as np

from pymol import cmd
from tqdm import tqdm


def create_distance_features_by_tcr(tcr='B11'):
    model_list = get_model_list(tcr)
    feature_matrix = get_feature_matrix(model_list)
    save_feature_matrix(feature_matrix, tcr, model_list)


def create_distance_features_by_epitope(epitope='SIINFEKL'):
    model_list = get_model_list(epitope)
    feature_matrix = get_feature_matrix(model_list)
    save_feature_matrix_string(feature_matrix, epitope, model_list)


def get_model_list(folder_name):
    path_base = f'Models/{folder_name}/'
    path_models = [path_base + file_name for file_name in os.listdir(path_base)]
    return path_models


def get_feature_matrix(model_list):
    feature_matrix = []
    for model in tqdm(model_list):
        feature_matrix.append(get_features_per_model(model))
    return feature_matrix


def get_features_per_model(path_model):
    cmd.load(path_model)
    features_beta = Dist.plot_distances(do_beta=True, do_save=False, do_return=True, mode='com', use_igtm=False)
    features_alpha = Dist.plot_distances(do_beta=False, do_save=False, do_return=True, mode='com', use_igtm=False)
    cmd.delete('all')
    return features_alpha, features_beta


def save_feature_matrix(features, name, model_list):
    columns = []
    for chain in ['a', 'b']:
        if chain == 'a':
            len_chain = features[0][0].shape[1]
        else:
            len_chain = features[0][1].shape[1]
        for i in range(1, 9):
            for j in range(1, len_chain+1):
                columns.append(f'{chain}_{i}_{j}')
    indices = [x.split('_')[-1][:-4] for x in model_list]
    features = [np.concatenate([x.flatten(), y.flatten()], axis=None) for x, y in features]
    df = pd.DataFrame(data=features, index=indices, columns=columns)
    path_out = f'Output/features/distances_{name}.csv'
    df.to_csv(path_out)


def save_feature_matrix_string(features, name, model_list):
    indices = [x.split('/')[-1].split('_')[0] for x in model_list]
    features_x = [','.join([str(el) for el in x.flatten()]) for x, _ in features]
    features_y = [','.join([str(el) for el in y.flatten()]) for _, y in features]
    df = pd.DataFrame(index=indices)
    df['features_alpha'] = features_x
    df['features_beta'] = features_y
    path_out = f'Output/features/distances_{name}.csv'
    df.to_csv(path_out)


if __name__ == '__main__':
    # create_distance_features_by_tcr()
    create_distance_features_by_epitope(epitope='EIINFEKL')