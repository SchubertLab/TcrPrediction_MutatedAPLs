import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

from pymol import cmd


aminos_3_to_1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}


def plot_distances(do_beta=True, do_save=False, do_return=False, mode='min', use_igtm=False, idx=0):
    epitope, chain = get_sequence_names(do_beta, idx=idx)
    epitope_sel = find_sequence(epitope)
    chain_sel = find_sequence(chain)

    distances_matrix = calculate_distance_matrix(epitope_sel, chain_sel, mode=mode)

    if do_return:
        return distances_matrix

    #plot_matrix(distances_matrix, list(chain), list(epitope), do_save=do_save, use_igtm=use_igtm, do_beta=do_beta,
    #idx=idx)
    if do_save:
        save_matrix(distances_matrix, do_beta=do_beta, idx=idx)


def get_sequence_names(do_beta=True, idx=0):
    tcr_name, epitope = cmd.get_names()[idx].split('_')
    chain_type = 'CDR3B'
    if not do_beta:
        chain_type = 'CDR3A'
    path_tcrs = '../data/TCR_info.csv'
    df_tcrs = pd.read_csv(path_tcrs, index_col=0)
    chain = df_tcrs[chain_type][tcr_name]
    return epitope, chain


def find_sequence(sequence):
    my_space = {'amino_list': [],
                'aminos_3_to_1': aminos_3_to_1}
    cmd.iterate("name ca", "amino_list.append((resi,aminos_3_to_1[resn],chain))", space=my_space)
    amino_list = my_space['amino_list']
    amino_seq = ''.join([x[1] for x in amino_list])
    idx_start = amino_seq.find(sequence)
    idx_end = idx_start + len(sequence)
    return amino_list[idx_start:idx_end]


def calculate_distance_matrix(epitope, chain, mode='min'):
    distance_functions = {
        'min': calculate_minimal_distances,
        'com': calculate_center_of_mass_distance,
    }
    distance_function = distance_functions[mode]

    distance_matrix = np.ones(shape=(len(epitope), len(chain))) * 999
    for i in range(len(epitope)):
        cmd.select('tmp_ep', f'resi {epitope[i][0]} in chain {epitope[i][2]}')
        for j in range(len(chain)):
            cmd.select('tmp_chain', f'res {chain[j][0]} in chain {chain[j][2]}')
            distance_matrix[i, j] = distance_function('tmp_ep', 'tmp_chain')
    return distance_matrix


def calculate_minimal_distances(selection_1, selection_2):
    min_dist = 99999
    for at1 in cmd.index(selection_1):
        for at2 in cmd.index(selection_2):
            dist = cmd.get_distance(at1, at2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def calculate_center_of_mass_distance(selection_1, selection_2):
    position_1 = cmd.centerofmass(selection_1)
    position_2 = cmd.centerofmass(selection_2)
    distance = [x[0]-x[1] for x in zip(position_1, position_2)]
    distance = [x*x for x in distance]
    distance = sum(distance)
    distance = math.sqrt(distance)
    return distance


def plot_matrix(matrix, x_ticks, y_ticks, do_save, use_igtm=False, do_beta=True, idx=0):
    if do_save:
        plt.clf()
    plt.figure(figsize=(15, 8))
    if use_igtm:
        x_ticks = sequence_to_igtm(''.join(x_ticks))

    ax = sns.heatmap(matrix, center=0, linewidths=0.5, cmap='RdBu_r', square=True,
                         xticklabels=x_ticks, vmax=30, vmin=0, annot=True, fmt=".1f")
    ax.set_yticklabels(y_ticks, rotation=0)

    if not do_save:
        plt.tight_layout()
        plt.show()
        return
    if do_beta:
        path_out = '../results/model3d/' + cmd.get_names()[idx] + '_beta'
    else:
        path_out = '../results/model3d/' + cmd.get_names()[idx] +'_alpha'
    plt.savefig(path_out, dpi='figure')


def save_matrix(matrix, do_beta, idx=0):
    if do_beta:
        path_out = '../results/model3d/' + cmd.get_names()[idx] + '_beta.csv'
    else:
        path_out = '../results/model3d/' + cmd.get_names()[idx] + '_alpha.csv'
    df = pd.DataFrame(matrix)
    df.to_csv(path_out)


def sequence_to_igtm(sequence):
    scheme = [str(x) for x in range(104, 112)]
    scheme += ['111.' + str(x) for x in range(1, 10)]
    scheme += ['112.' + str(x) for x in reversed(range(1, 10))]
    scheme += [str(x) for x in range(112, 119)]

    front = []
    back = []
    for i in range(len(sequence)):
        if i % 2 == 0:
            front.append(scheme[i//2])
        else:
            back.insert(0, scheme[-i//2])
    # return scheme[: (len(sequence)+1)//2] + scheme[-len(sequence)//2:]
    return front + back

amino_list = []


if __name__ == 'pymol':
    plot_distances(do_beta=False, do_save=True, mode='com', use_igtm=True, idx=0)
    plot_distances(do_beta=True, do_save=True, mode='com', use_igtm=True, idx=0)
    print('finished')
