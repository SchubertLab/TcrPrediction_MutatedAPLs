import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

# from pymol import cmd


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


def plot_distances(do_beta=True, do_save=False, mode='min'):
    epitope, chain = get_sequence_names(do_beta)
    epitope_sel = find_sequence(epitope)
    chain_sel = find_sequence(chain)

    distances_matrix = calculate_distance_matrix(epitope_sel, chain_sel, mode=mode)
    plot_matrix(distances_matrix, list(chain), list(epitope), do_save=do_save)


def get_sequence_names(do_beta=True):
    tcr_name, epitope = cmd.get_names()[0].split('_')
    chain_type = 'CDR3B'
    if not do_beta:
        chain_type = 'CDR3A'
    path_tcrs = '../data/TCR_info.csv'
    df_tcrs = pd.read_csv(path_tcrs, index_col=0)
    chain = df_tcrs[chain_type][tcr_name]
    return epitope, chain


def find_sequence(sequence):
    cmd.iterate("name ca", "amino_list.append((resi,aminos_3_to_1[resn],chain))")
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


def plot_matrix(matrix, x_ticks, y_ticks, do_save, do_igtm=False):
    plt.clf()
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(matrix, center=0, linewidths=0.5, cmap='RdBu_r', square=True,
                         xticklabels=x_ticks, vmax=30, vmin=0, annot=True, fmt=".1f")
    ax.set_yticklabels(y_ticks, rotation=0)

    if not do_save:
        plt.tight_layout()
        plt.show()
        return
    path_out = '../results/model3d/' + cmd.get_names()[0]
    plt.savefig(path_out, dpi='figure')


def sequence_to_igtm(sequence):
    print((len(sequence)+1)//2)
    scheme = [str(x) for x in range(104, 112)]
    scheme += ['111.' + str(x) for x in range(1, 10)]
    scheme += ['112.' + str(x) for x in reversed(range(1, 10))]
    scheme += [str(x) for x in range(112, 119)]
    return scheme[: (len(sequence)+1)//2] + scheme[-len(sequence)//2:]


amino_list = []
# plot_distances(do_beta=True, do_save=True, mode='com')
# print(list('CASSRANYEQYF'))
print(sequence_to_igtm('CASSRANYEQYF'))
# print('---Finished---')
