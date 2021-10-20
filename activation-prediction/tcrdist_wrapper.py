import pandas as pd
from tcrdist.repertoire import TCRrep


tcr_info = pd.read_csv('../data/TCR_info.csv')

rename_dict = {
    'TRBV': 'v_b_gene',
    'TRBD': 'd_b_gene',
    'TRBJ': 'j_b_gene',
    'CDR3B': 'cdr3_b_aa',
    'TRAV': 'v_a_gene',
    'TRAJ': 'j_a_gene',
    'CDR3A': 'cdr3_a_aa'
}
tcr_info = tcr_info.rename(columns=rename_dict)

tr = TCRrep(cell_df=tcr_info, organism='mouse', chains=['beta', 'alpha'],
            db_file='alphabeta_gammadelta_db.tsv')
distances = tr.pw_beta

tcrs = tcr_info['name'].tolist()
df_distances = pd.DataFrame(data=distances, columns=tcrs, index=tcrs)
df_distances.to_csv('results/tcrdist_all_tcrs.csv')