import numpy as np
import pandas as pd
import ast
import os


path_base = os.path.dirname(__file__)


def start_padding(matrix, length, tcr):
    padding = np.ones(shape=(8, length)) * -99
    padding[:, -matrix.shape[1]:] = matrix
    return padding


def end_padding(matrix, length, tcr):
    padding = np.ones(shape=(8, length)) * -99
    padding[:, :matrix.shape[1]] = matrix
    return padding


def middle_padding(matrix, length, tcr):
    padding = np.ones(shape=(8, length)) * -99
    padding[:, :int(matrix.shape[1]/2)] = matrix[:, :int(matrix.shape[1]/2)]
    padding[:, -int(matrix.shape[1]/2):] = matrix[:, -int(matrix.shape[1]/2):]
    return padding


def muscle_padding(matrix, length, tcr_info):
    tcr, do_alpha = tcr_info
    muscle_coding = read_muscle_coding(tcr, do_alpha)
    padding = np.ones(shape=(8, len(muscle_coding))) * -99

    idx_matrix = 0
    for idx_padding, letter in enumerate(muscle_coding):
        if letter != '-':
            padding[:, idx_padding] = matrix[:, idx_matrix]
            idx_matrix += 1
    return padding


def read_muscle_coding(tcr, do_alpha):
    current_tcr = ''
    encoding = None
    if do_alpha:
        fasta_file = open(path_base + '/data/cdr3a-aligned.fasta', 'r')
    else:
        fasta_file = open(path_base + '/data/cdr3b-aligned.fasta', 'r')
    while current_tcr.lower() != tcr.lower():
        current_tcr = fasta_file.readline()[1:-1]
        current_tcr = current_tcr.replace('_', '')
        encoding = fasta_file.readline()
    fasta_file.close()
    encoding = encoding.replace(' ', '')
    encoding = encoding.replace('\n', '')
    return encoding


ENCODING_METHOD = 'middle'
padding_functions = {
    'end': end_padding,
    'start': start_padding,
    'middle': middle_padding,
    'muscle': muscle_padding,
}


def get_distances(epitope='SIINFEKL'):
    data = pd.read_csv(f'{path_base}/Modelling3D/Output/features/distances_{epitope}.csv', index_col=0)

    features_alpha = features_from_string(data, do_alpha=True)
    columns_alpha = get_column_names(features_alpha.shape[1], 'a')
    df_alpha = pd.DataFrame(data=features_alpha, columns=columns_alpha)
    df_alpha['tcr'] = data.index
    df_alpha = df_alpha.set_index('tcr')

    features_beta = features_from_string(data, do_alpha=False)
    columns_beta = get_column_names(features_beta.shape[1], 'b')
    df_beta = pd.DataFrame(data=features_beta, columns=columns_beta)
    df_beta['tcr'] = data.index
    df_beta = df_beta.set_index('tcr')

    data = pd.concat([data, df_alpha, df_beta], axis=1)
    data = data.drop(['features_alpha', 'features_beta'], axis=1)
    data = data.sort_index()

    return data


def get_column_names(n, prefix):
    length_chain = int(n/8)
    names = []
    for i in range(n):
        pos_epitope = i // length_chain
        pos_chain = i % length_chain
        name = f'{prefix}_{pos_epitope}_{pos_chain}'
        names.append(name)
    return names


def features_from_string(feature, do_alpha):
    tcrs = feature.index
    if do_alpha:
        feature_string = feature['features_alpha']
    else:
        feature_string = feature['features_beta']
    matrix = [np.array(ast.literal_eval('[' + x + ']')) for x in feature_string.values]
    matrix = [np.reshape(x, newshape=(8, int(x.shape[0]/8))) for x in matrix]
    max_length = max([x.shape[1] for x in matrix])
    matrix_padded = []
    for mat, tcr in zip(matrix, tcrs):
        padded = padding_functions[ENCODING_METHOD](mat, max_length, (tcr, do_alpha))
        matrix_padded.append(padded)
    matrix_padded = [x.flatten() for x in matrix_padded]
    return np.stack(matrix_padded, axis=0)


def get_labels(tcrs, epitope='SIINFEKL', do_binary=True):
    activations_old = pd.read_csv(path_base + '/data/activations_lena.csv', index_col=0)
    activations_old = activations_old.transpose()
    activations_old['is_activated'] = activations_old[epitope] > 15

    activations_new = pd.read_csv(path_base + '/data/activations_phillip.csv', index_col=0)
    activations_new = activations_new.transpose()
    activations_new['is_activated'] = activations_new[epitope] > 15
    activations_new = activations_new.drop(['OT1'])

    activations = pd.concat([activations_old, activations_new])
    activations = activations[activations.index.isin(tcrs)]
    activations = activations.sort_index()
    if do_binary:
        return activations[['is_activated']]
    return activations[[epitope]]
