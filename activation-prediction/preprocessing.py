#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# %%


def read_fasta(fname):
    with open(fname) as f:
        pname = None
        prots = {}
        for row in f:
            row = row.strip()
            if not row:
                continue

            if row[0] == '>':
                if pname is not None:
                    prots[pname] = ''.join(prots[pname])
                pname = row[1:]

                prots[pname] = []
            else:
                prots[pname].append(row)
    if pname is not None:
        prots[pname] = ''.join(prots[pname])
    return prots


def get_cdr_sequences(alignment_type='muscle'):
    # naive repertoire
    ntdf = pd.read_csv(
        '../data/naive repertoire SIINFEKL APL landscape data V2 TCR INFO.csv'
    )
    ntdf['tcr'] = ntdf['sample id (TCR)'].apply(
        lambda s: s.split('_')[3] if '_' in s else s.replace('-', '')
    )

    ntdf = ntdf.rename(columns={
        'CDR3α': 'cdr3a',
        'CDR3β': 'cdr3b',
    })[['tcr', 'cdr3a', 'cdr3b']]

    # educated repertoire
    etdf = pd.read_excel(
        '../data/Educated_repertoire_derived_TCRs_CDR3a_b_TCRa_b.xlsx'
    )[[
        'TCR', 'CDR3b', 'CDR3a'
    ]].rename(columns={
        'TCR': 'tcr', 'CDR3b': 'cdr3b', 'CDR3a': 'cdr3a'
    })
    etdf['tcr'] = etdf['tcr'].str.upper()

    # alignment
    if alignment_type == 'muscle':
        cadf = pd.DataFrame(read_fasta('../data/cdr3a-aligned.fasta').items(),
                            columns=['tcr', 'cdr3a_aligned'])
    
        cbdf = pd.DataFrame(read_fasta('../data/cdr3b-aligned.fasta').items(),
                            columns=['tcr', 'cdr3b_aligned'])
    elif alignment_type == 'imgt':
        cadf = pd.DataFrame(read_fasta('../data/cdr3a-aligned-imgt.fasta').items(),
                            columns=['tcr', 'cdr3a_aligned'])
    
        cbdf = pd.DataFrame(read_fasta('../data/cdr3b-aligned-imgt.fasta').items(),
                            columns=['tcr', 'cdr3b_aligned'])
    else:
        raise ValueError('unknown CDR3 alignment (valid choices: muscle, imgt)')

    # merge and return
    tdf = pd.concat([ntdf, etdf]).merge(cadf, how='outer').merge(cbdf, how='outer')

    # create copies for other OT1 with different names
    # TODO check that these OT1 are actually all the same

    tdf.loc[-1] = tdf.query('tcr=="OT1"').iloc[0]
    tdf.loc[-1, 'tcr'] = 'OTI_PH'

    tdf.loc[-2] = tdf.query('tcr=="OT1"').iloc[0]
    tdf.loc[-2, 'tcr'] = 'LR_OTI_1'

    tdf.loc[-3] = tdf.query('tcr=="OT1"').iloc[0]
    tdf.loc[-3, 'tcr'] = 'LR_OTI_2'

    return tdf.reset_index(drop=True)


def get_dataset(educated_repertoire=None, normalization=None,
                cdr3_alignment_type='muscle'):
    if educated_repertoire is None:
        return pd.concat([
            get_dataset(True, normalization, cdr3_alignment_type),
            get_dataset(False, normalization, cdr3_alignment_type),
        ]).reset_index(drop=True)

    if educated_repertoire:
        fname = '../data/PH_data_educated_repertoire_and_OTI.xlsx'
    else:
        fname = '../data/LR_data_naive_repertoire_and_OTI.xlsx'

    if normalization is None:
        sheet = 'Unnormalized Data'
    elif normalization == 'AS':
        sheet = 'Normalized to initial AS'
    elif normalization == 'OT1':
        sheet = 'Normalized to internal OTI'
    else:
        raise ValueError('unknown normalization')

    df = pd.read_excel(fname, sheet)

    # unpack the columns into each tcr
    dds = []
    for i in range(0, len(df.columns), 4):
        d = df.iloc[:, [i, i + 3]].copy()
        d.columns = ['apl', 'activation']
        d['tcr'] = df.columns[i].upper()
        dds.append(d)
    df = pd.concat(dds)

    # convert apl to mutation position and amino acid
    apl_to_mut = {
        i + 1: (p, a)
        for i, (p, a) in enumerate([
            (p, a)
            for p in range(8)
            for a in 'ACDEFGHIKLMNPQRSTVWY'
            if a != 'SIINFEKL'[p]
        ])
    }

    apl_to_epi = {
        i: ''.join(
            b if j != p else a
            for j, b in enumerate('SIINFEKL')
        )
        for i, (p, a) in apl_to_mut.items()
    }

    df['mut_pos'] = df['apl'].apply(lambda x: apl_to_mut[x][0] if x in apl_to_mut else -1)
    df['mut_ami'] = df['apl'].apply(lambda x: apl_to_mut[x][1] if x in apl_to_mut else None)
    df['orig_ami'] = df['mut_pos'].apply(lambda x: 'SIINFEKL'[x])
    df['epitope'] = df['apl'].apply(lambda x: apl_to_epi.get(x, 'SIINFEKL'))
    df = df.drop(columns='apl')

    # add cdr sequences
    df = df.merge(get_cdr_sequences(cdr3_alignment_type), on='tcr', how='left')

    # compute residual
    df = df.merge(
        # wild-type activation for each tcr
        df.query('mut_pos < 0')[[
            'tcr', 'activation'
        ]].rename(columns={
            'activation': 'wild_activation'
        }),
        on='tcr'
    )
    df['residual'] = df['activation'] - df['wild_activation']

    df['normalization'] = normalization or 'none'

    return df.reset_index(drop=True)


def get_complete_dataset():
    dfs = []

    for edu in [True, False]:
        for norm in [None, 'AS', 'OT1']:
            df = get_dataset(edu, norm)
            df['is_educated'] = edu
            dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def add_activation_thresholds(df, key='is_activated', remove_constant=True):
    ds = []
    for _, g in df.groupby('normalization'):
        for thr in [15, 46.9, 66.8]:
            g['threshold'] = str(thr)
            g[key] = g['activation'] > thr
            ds.append(g.copy())

    res = pd.concat(ds)
    return res.groupby([
        'tcr', 'normalization', 'threshold'
    ]).filter(
        lambda g: 0 < g[key].sum() < len(g)
    ).reset_index(drop=True) if remove_constant else res.reset_index(drop=True)


# %% amino acid features
def get_aa_factors_2():
    # table 1 in 10.1007/s00894-001-0058-5
    table = '''A 0.008 0.134 –0.475 –0.039 0.181
R 0.171 –0.361 0.107 –0.258 –0.364
N 0.255 0.038 0.117 0.118 –0.055
D 0.303 –0.057 –0.014 0.225 0.156
C –0.132 0.174 0.070 0.565 –0.374
Q 0.149 –0.184 –0.030 0.035 –0.112
E 0.221 –0.280 –0.315 0.157 0.303
G 0.218 0.562 –0.024 0.018 0.106
H 0.023 –0.177 0.041 0.280 –0.021
I –0.353 0.071 –0.088 –0.195 –0.107
L –0.267 0.018 –0.265 –0.274 0.206
K 0.243 –0.339 –0.044 –0.325 –0.027
M –0.239 –0.141 –0.155 0.321 0.077
F –0.329 –0.023 0.072 –0.002 0.208
P 0.173 0.286 0.407 –0.215 0.384
S 0.199 0.238 –0.015 –0.068 –0.196
T 0.068 0.147 –0.015 –0.132 –0.274
W –0.296 –0.186 0.389 0.083 0.297
Y –0.141 –0.057 0.425 –0.096 –0.091
V –0.274 0.136 –0.187 –0.196 –0.299'''.replace('–', '-')

    aa_factors = {
        aa: [float(f) for f in factors]
        for row in table.split('\n')
        for aa, *factors in [row.split()]
    }

    aa_facs = pd.DataFrame(aa_factors)
    aa_facs['feature_category'] = 'factors2'
    aa_facs['feature'] = [f'factor{i}' for i in range(len(aa_facs))]
    aa_facs = aa_facs.set_index(['feature_category', 'feature']).T

    return aa_facs


def get_aa_factors():
    # table 2 in https://www.pnas.org/content/102/18/6395
    aa_factors_table = '''A 	−0.591 	−1.302 	−0.733 	1.570 	−0.146
    C 	−1.343 	0.465 	−0.862 	−1.020 	−0.255
    D 	1.050 	0.302 	−3.656 	−0.259 	−3.242
    E 	1.357 	−1.453 	1.477 	0.113 	−0.837
    F 	−1.006 	−0.590 	1.891 	−0.397 	0.412
    G 	−0.384 	1.652 	1.330 	1.045 	2.064
    H 	0.336 	−0.417 	−1.673 	−1.474 	−0.078
    I 	−1.239 	−0.547 	2.131 	0.393 	0.816
    K 	1.831 	−0.561 	0.533 	−0.277 	1.648
    L 	−1.019 	−0.987 	−1.505 	1.266 	−0.912
    M 	−0.663 	−1.524 	2.219 	−1.005 	1.212
    N 	0.945 	0.828 	1.299 	−0.169 	0.933
    P 	0.189 	2.081 	−1.628 	0.421 	−1.392
    Q 	0.931 	−0.179 	−3.005 	−0.503 	−1.853
    R 	1.538 	−0.055 	1.502 	0.440 	2.897
    S 	−0.228 	1.399 	−4.760 	0.670 	−2.647
    T 	−0.032 	0.326 	2.213 	0.908 	1.313
    V 	−1.337 	−0.279 	−0.544 	1.242 	−1.262
    W 	−0.595 	0.009 	0.672 	−2.128 	−0.184
    Y 	0.260 	0.830 	3.097 	−0.838 	1.512'''.replace('−', '').replace(' ', '')

    aa_factors = {
        aa: [float(f) for f in factors]
        for row in aa_factors_table.split('\n')
        for aa, *factors in [row.split('\t')]
    }

    aa_facs = pd.DataFrame(aa_factors)
    aa_facs['feature_category'] = 'factors'
    aa_facs['feature'] = [f'factor{i}' for i in range(len(aa_facs))]
    aa_facs = aa_facs.set_index(['feature_category', 'feature']).T

    return aa_facs


def get_aa_blosum():
    blosum_str = '''#  Matrix made by matblas from blosum62.iij
#  * column uses minimum score
#  BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
#  Blocks Database = /data/blocks_5.0/blocks.dat
#  Cluster Percentage: >= 62
#  Entropy =   0.6979, Expected =  -0.5209
#  Downloaded from https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt 25/11/2020
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1'''

    aminos = None
    blosum = {}
    for row in blosum_str.split('\n'):
        if row.startswith('#'):
            continue

        if aminos is None:
            aminos = row.split()
        else:
            a, *scores = row.split()
            blosum[a] = [int(s) for s in scores]

    bs = pd.DataFrame(blosum)
    bs['feature_category'] = 'blosum'
    bs['feature'] = [f'blosum_{a}' for a in aminos]
    bs = bs.set_index(['feature_category', 'feature']).T

    return bs


def get_aa_chem():
    # https://en.wikipedia.org/wiki/Amino_acid
    amino_props = '''Alanine 	Ala 	A 	Aliphatic 	Nonpolar 	Neutral 	1.8 			89.094 	8.76 	GCN
    Arginine 	Arg 	R 	Basic 	Basic polar 	Positive 	−4.5 			174.203 	5.78 	MGR, CGY (coding codons can also be expressed by: CGN, AGR)
    Asparagine 	Asn 	N 	Amide 	Polar 	Neutral 	−3.5 			132.119 	3.93 	AAY
    Aspartic acid 	Asp 	D 	Acid 	Acidic polar 	Negative 	−3.5 			133.104 	5.49 	GAY
    Cysteine 	Cys 	C 	Sulfuric 	Nonpolar 	Neutral 	2.5 	250 	0.3 	121.154 	1.38 	UGY
    Glutamine 	Gln 	Q 	Amide 	Polar 	Neutral 	−3.5 			146.146 	3.9 	CAR
    Glutamic acid 	Glu 	E 	Acid 	Acidic polar 	Negative 	−3.5 			147.131 	6.32 	GAR
    Glycine 	Gly 	G 	Aliphatic 	Nonpolar 	Neutral 	−0.4 			75.067 	7.03 	GGN
    Histidine 	His 	H 	Basic aromatic 	Basic polar 	Positive, 10% Neutral, 90% 	−3.2 	211 	5.9 	155.156 	2.26 	CAY
    Isoleucine 	Ile 	I 	Aliphatic 	Nonpolar 	Neutral 	4.5 			131.175 	5.49 	AUH
    Leucine 	Leu 	L 	Aliphatic 	Nonpolar 	Neutral 	3.8 			131.175 	9.68 	YUR, CUY (coding codons can also be expressed by: CUN, UUR)
    Lysine 	Lys 	K 	Basic 	Basic polar 	Positive 	−3.9 			146.189 	5.19 	AAR
    Methionine 	Met 	M 	Sulfuric 	Nonpolar 	Neutral 	1.9 			149.208 	2.32 	AUG
    Phenylalanine 	Phe 	F 	Aromatic 	Nonpolar 	Neutral 	2.8 	257, 206, 188 	0.2, 9.3, 60.0 	165.192 	3.87 	UUY
    Proline 	Pro 	P 	Cyclic 	Nonpolar 	Neutral 	−1.6 			115.132 	5.02 	CCN
    Serine 	Ser 	S 	Hydroxylic 	Polar 	Neutral 	−0.8 			105.093 	7.14 	UCN, AGY
    Threonine 	Thr 	T 	Hydroxylic 	Polar 	Neutral 	−0.7 			119.119 	5.53 	ACN
    Tryptophan 	Trp 	W 	Aromatic 	Nonpolar 	Neutral 	−0.9 	280, 219 	5.6, 47.0 	204.228 	1.25 	UGG
    Tyrosine 	Tyr 	Y 	Aromatic 	Polar 	Neutral 	−1.3 	274, 222, 193 	1.4, 8.0, 48.0 	181.191 	2.91 	UAY
    Valine 	Val 	V 	Aliphatic 	Nonpolar 	Neutral 	4.2 			117.148 	6.73 	GUN'''.replace('−', '-')

    aa_props = pd.DataFrame([
        [q.strip() for q in x.split('\t')]
        for x in amino_props.split('\n')
    ]).drop(columns=[0, 1, 7, 8, 11]).rename(columns={
        2: 'aa', 3: 'class', 4: 'polarity', 5: 'charge', 6: 'hydropathy', 9: 'mass', 10: 'abundance'
    }).set_index('aa')

    for col in ['hydropathy', 'mass', 'abundance']:
        aa_props[col] = aa_props[col].astype(np.float64)

    aa_props = pd.get_dummies(aa_props)
    aa_props = aa_props.T.reset_index().rename(columns={'index': 'feature'})
    aa_props['feature_category'] = 'chem'
    aa_props = aa_props.set_index(['feature_category', 'feature']).T

    return aa_props


def make_1hot(i, n):
    res = [0] * n
    res[i] = 1
    return res


def get_aa_1hot():
    aa_1hot = {a: make_1hot(i, 20) for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    aa1h = pd.DataFrame(aa_1hot)
    aa1h['feature_category'] = 'one_hot'
    aa1h['feature'] = [f'is_{a}' for a in 'ACDEFGHIKLMNPQRSTVWY']
    aa1h = aa1h.set_index(['feature_category', 'feature']).T
    return aa1h


def get_aa_features():
    fs = pd.concat([
        get_aa_factors_2(),
        get_aa_factors(),
        get_aa_blosum(),
        get_aa_chem(),
        get_aa_1hot()
    ], axis=1)

    # add additional row for alignment gaps
    fs['one_hot', 'is_gap'] = 0
    fs.loc['-'] = pd.Series(
        [0] * len(fs.columns),
        index=fs.columns
    )
    fs.loc['-', ('one_hot', 'is_gap')] = 1

    return fs

# %% feature extraction


class FeatureMaker:
    # feature names are taken from the *first* complete sample
    def __init__(self):
        self._names = []
        self._samples = []
        self._n_feats = None

    def new_sample(self):
        if len(self._samples) == 1:
            self._n_feats = len(self._samples[0])
            assert len(self._samples[0]) == len(self._names)
        elif len(self._samples) >= 1:
            assert len(self._samples[-1]) == self._n_feats

        self._samples.append([])

    def add_sample_features(self, name, value):
        add_to_names = self._n_feats is None
        if isinstance(value, pd.Series):
            self._samples[-1].extend(value.tolist())
            if add_to_names:
                self._names.extend([
                    name + '$' + (
                        '$'.join(map(str, x)) if isinstance(x, (list, tuple))
                        else str(x)
                    )
                    for x in value.index
                ])
        elif isinstance(value, (list, tuple)):
            self._samples[-1].extend(value)
            if add_to_names:
                self._names.extend([f'{name}${i}' for i in range(len(value))])
        else:
            self._samples[-1].append(float(value))
            if add_to_names:
                self._names.append(name)

    def get_dataset(self):
        return pd.DataFrame(self._samples, columns=self._names)


def full_aa_features(fit_data, aa_features, interactions=False,
                     include_tcr=False, include_mutation=True,
                     remove_constant=False):

    fs = aa_features
    if interactions:
        pf = PolynomialFeatures().fit_transform(aa_features)
        fs = pd.DataFrame(pf, index=aa_features.index)

    feats = FeatureMaker()
    for _, row in fit_data.iterrows():
        feats.new_sample()

        # mutation data
        if include_mutation:
            feats.add_sample_features('mut_pos', row['mut_pos'])
            feats.add_sample_features('mut_ami', fs.loc[row['mut_ami']])
            feats.add_sample_features('orig_ami', fs.loc[row['orig_ami']])

        # whole epitope
        for i, a in enumerate(row['epitope'].strip()):
            feats.add_sample_features(f'epi_{i}', fs.loc[a])

        # wild type
        for i, a in enumerate('SIINFEKL'):
            feats.add_sample_features(f'wild_{i}', fs.loc[a])

        # difference between the two
        for i, (a1, a2) in enumerate(zip('SIINFEKL', row['epitope'].strip())):
            feats.add_sample_features(f'diff_{i}', fs.loc[a1] - fs.loc[a2])

        # aligned TCRs
        if include_tcr:
            for i, a in enumerate(row['cdr3a_aligned'].strip()):
                feats.add_sample_features(f'cdr3a_{i}', fs.loc[a])

            for i, a in enumerate(row['cdr3b_aligned'].strip()):
                feats.add_sample_features(f'cdr3b_{i}', fs.loc[a])

    res = feats.get_dataset()
    res.index = fit_data.index

    # remove constant columns
    if remove_constant:
        res = res.loc[:, res.std(axis=0) > 1e-6]

    return res


# %% grouping

def build_feature_groups(columns):
    groups = set([c.split('$')[0] for c in columns])
    feature_groups = {'all': []}

    for i in range(8):
        feature_groups[f'pos_{i}'] = [f'wild_{i}', f'epi_{i}', f'diff_{i}']
        groups.difference_update(feature_groups[f'pos_{i}'])

    for g in groups:
        feature_groups[g] = [g]

    feature_groups['cdr3'] = [c for c in columns if c.startswith('cdr3')]
    feature_groups['cdr3a'] = [c for c in columns if c.startswith('cdr3a')]
    feature_groups['cdr3b'] = [c for c in columns if c.startswith('cdr3b')]

    return feature_groups


def decorrelate_groups(fit_data, groups, keep_var=0.99):
    # applies pca independently to the features in each group
    # and only keeps the top N pc so that the total variance is
    # as indicated
    all_reduced = []
    for g, cs in groups.items():
        if g == 'all':
            continue

        keep_columns = [c for c in fit_data.columns if c.split('$')[0] in cs]
        data = fit_data[keep_columns]

        pca = PCA().fit(data)
        n = np.sum(np.cumsum(pca.explained_variance_ratio_) <= keep_var)
        if n > 0:
            all_reduced.append(pd.DataFrame(
                pca.transform(data)[:, :n],
                columns=[f'{g}${i}' for i in range(n)]
            ))

    res = pd.concat(all_reduced, axis=1)
    res.index = fit_data.index
    return res
