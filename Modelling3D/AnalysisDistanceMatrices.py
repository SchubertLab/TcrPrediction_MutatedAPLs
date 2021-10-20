import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import utils_distance_matrix as Utils


def get_dataset():
    data = pd.read_csv('Output/features/distances_B11.csv', index_col=0)
    data = data.sort_index()
    return data


def get_labels():
    activations = pd.read_csv('../data/activations_lena.csv', index_col=0)
    activations = activations.sort_index()
    activations['is_activated'] = activations['B11'] > 15
    return activations


def get_dataset_across_tcr(epitope):
    data = Utils.get_distances(epitope)
    print(data)
    raise ValueError
    label = Utils.get_labels(data.index, epitope, do_binary=False)
    return data, label


def plot_stats_across_tcr(epitope='SIINFEKL'):
    data, labels = get_dataset_across_tcr(epitope)
    print(data.columns)
    correlation(data, labels.values, 'SIINFEKL', length_beta=16)

    print(data.head())
    print(labels.shape)
    # print(np.min(data.values))


def plot_stats_per_epitope_position():
    data = get_dataset()
    data = data[data.index != 'SIINFEKL']
    mutations = get_mutated_position(data.index)
    labels = get_labels()
    labels = labels[labels.index != 'SIINFEKL']
    statistics_per_pos(data, mutations)
    correlation_per_pos(data, labels, mutations)


def get_mutated_position(epitopes):
    base = 'SIINFEKL'
    positions = []
    for mutation in epitopes:
        for idx, (letter_mutation, letter_base) in enumerate(zip(mutation, base)):
            if letter_mutation != letter_base:
                positions.append(idx)
                break
    return positions


def statistics_per_pos(data, mutations):
    mean = apply_function_per_pos(np.mean, mutations, data)
    std = apply_function_per_pos(np.std, mutations, data)
    plot_to_matrix(mean, 14, data.columns, 'Output/figs/B11_mean_per_pos')
    plot_to_matrix(std, 14, data.columns, 'Output/figs/B11_std_per_pos')


def apply_function_per_pos(func, mutations, data):
    output = np.ones(data.values.shape[1]) * -99
    for pos in range(8):
        subset = data[[x == pos for x in mutations]]
        sub_result = func(subset.values, axis=0)
        for idx, name in enumerate(data.columns):
            epitope_pos = int(name.split('_')[1]) - 1
            if epitope_pos == pos:
                output[idx] = sub_result[idx]
    return output


def correlation_per_pos(data, labels, mutations):
    labels = labels['B11']
    n = data.shape[1]
    corr_values = np.ones(shape=(n,)) * -99
    for idx, name in enumerate(data.columns):
        position = int(name.split('_')[1]) - 1
        mask_positions = [x == position for x in mutations]
        data_subset = data[mask_positions]
        label_subset = labels[mask_positions]
        corr_values[idx] = np.corrcoef(label_subset.values, data_subset.values[:, idx])[0][1]
    plot_to_matrix(corr_values, 14, data.columns, 'Output/figs/B11_corr_per_pos')


def plot_stats():
    data = get_dataset()
    labels = get_labels()['B11'].values
    # statistics(data)
    correlation(data, labels, 'B11')


def statistics(data):
    mean = np.mean(data.values, axis=0)
    std = np.std(data.values, axis=0)
    plot_to_matrix(mean, 14, data.columns, 'Output/figs/B11_mean')
    plot_to_matrix(std, 14, data.columns, 'Output/figs/B11_std')


def correlation(data, labels, name, length_beta=14):
    labels = labels.flatten()
    n = data.shape[1]
    corr_values = np.ones(shape=(n,)) * -99
    for i in range(n):
        mask_gap = data.values[:, i] != -99
        label_sub = labels[mask_gap]
        data_sub = data.values[:, i][mask_gap]
        next_corr = np.corrcoef(label_sub, data_sub)[0][1]
        if math.isnan(next_corr):
            next_corr = 0.
        corr_values[i] = next_corr
    plot_to_matrix(corr_values, length_beta, data.columns, f'Output/figs/{name}_corr')


def plot_to_matrix(vector, length_beta, columns, path_base=None):
    matrix_beta = np.ones(shape=(8, length_beta)) * (-99)
    matrix_alpha = np.ones(shape=(8, vector.shape[0]//8-length_beta)) * (-99)

    for value, col in zip(vector, columns):
        matrix = matrix_beta
        if col[0] == 'a':
            matrix = matrix_alpha
        #print(col)
        position_epitope = int(col.split('_')[1]) - 1
        position_chain = int(col.split('_')[2]) - 1
        matrix[position_epitope, position_chain] = value
    vmax = max(np.max(matrix_alpha), np.max(matrix_beta))
    vmin = min(np.min(matrix_alpha), np.min(matrix_beta), 0)

    full_sequence = length_beta == 15

    heatmap(matrix_beta, path_out=path_base+'_beta.png', vmax=vmax, vmin=vmin, full_sequence=full_sequence)
    heatmap(matrix_alpha, is_beta=False, path_out=path_base+'_alpha.png', vmax=vmax, vmin=vmin, full_sequence=full_sequence)


def heatmap(matrix, is_beta=True, path_out=None, vmax=None, vmin=None, full_sequence=False):
    plt.figure(figsize=(15, 8))
    if not full_sequence:
        x_ticks = list(range(104, 119))
        x_ticks.remove(111)
        if not is_beta:
            x_ticks.remove(110)
            x_ticks.remove(112)
            x_ticks.remove(113)
    if is_beta:
        x_ticks = list(range(1, 17))
    else:
        x_ticks = list(range(1, 16))
    y_ticks = list(range(1, 9))
    ax = sns.heatmap(matrix, center=0, linewidths=0.5, cmap='RdBu_r', square=True,
                     vmax=vmax, vmin=vmin, annot=False, fmt=".1f")

    ax.set_xticklabels(x_ticks, rotation=-45)
    ax.set_yticklabels(y_ticks, rotation=0)
    if is_beta:
        ax.set(xlabel='CDR3β', ylabel='Epitope')
    else:
        ax.set(xlabel='CDR3α', ylabel='Epitope')
    if path_out is not None:
        plt.savefig(path_out)
    else:
        plt.show()


if __name__ == '__main__':
    # plot_stats()
    # plot_stats_per_epitope_position()
    plot_stats_across_tcr(epitope='SIINFEKL')
    # beta 'CASSGGTGRNTLYF'
    # alpha 'CAAGSNYQLIW'