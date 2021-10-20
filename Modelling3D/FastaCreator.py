import os
import pandas as pd


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def create_fasta_file(tcr, position=None, amino_acid_code=None, epitope=None):
    tcr_alpha, tcr_beta = read_tcrs(tcr)
    if epitope is None:
        epitope = get_epitope(position, amino_acid_code)
    sequences = {
        'MHC': read_mhc(),
        'TCRA': tcr_alpha,
        'TCRB': tcr_beta,
        'AG': epitope
    }
    path_out = f'fasta_files/{epitope}/{tcr}_{epitope}.fasta'
    write_to_file(path_out, sequences)


def read_mhc():
    path_mhc = '../data/h2-kb.txt'
    with open(path_mhc, 'r') as reader:
        reader.readline()
        mhc = reader.readline()
    return mhc


def read_tcrs(tcr_name):
    path_tcrs = '../data/tcrs_full_sequences.csv'
    df_tcrs = pd.read_csv(path_tcrs, index_col=0)
    alpha = df_tcrs['TCRA'][tcr_name]
    beta = df_tcrs['TCRB'][tcr_name]
    return alpha, beta


def get_epitope(position, amino_acid):
    base = 'SIINFEKL'
    if position is None or amino_acid is None:
        return base
    if isinstance(amino_acid, str):
        letter = amino_acid
    else:
        letter = AMINO_ACIDS[amino_acid]
    epitope = base[:position-1] + letter + base[position:]
    return epitope


def write_to_file(path_out, content):
    text = []
    for tag, sequence in content.items():
        text.append(f'>{tag}\n')
        text.append(f'{sequence.upper()}\n')
    with open(path_out, 'w+') as writer:
        writer.writelines(text)


if __name__ == '__main__':
    # for tcr in ['B11', 'B15', 'OT1']:
    #     for amino in ['F', 'D', 'L', 'K', None]:
    #         create_fasta_file(tcr, 3, amino)
    # create_fasta_file('B11', epitope='SIINAEKL')  # max in B11
    # create_fasta_file('B11', epitope='SIIVFEKL')  # min in B11

    # create_fasta_file('B15', epitope='RIINFEKL')  # max in B15
    # create_fasta_file('B15', epitope='SIIMFEKL')  # min in B15

    # create_fasta_file('OT1', epitope='SIINFERL')  # max in OT1
    # create_fasta_file('OT1', epitope='SIINFIKL')  # min in OT1

    # create_fasta_file('E4', epitope='SIINFEKL')
    # create_fasta_file('B10', epitope='SIINFEKL')
    base_epitope = 'SIIGFEKL'
    # create_fasta_file('B11', epitope=base_epitope)

    # for i in range(len(base_epitope)):
    #     for aa in AMINO_ACIDS:
    #         epitope = base_epitope[:i] + aa + base_epitope[i+1:]
    #         create_fasta_file('B11', epitope=epitope)
    df_tcrs = pd.read_csv('../data/tcrs_full_sequences.csv', index_col=0)
    for tcr in df_tcrs.index:
        create_fasta_file(tcr, epitope=base_epitope)
