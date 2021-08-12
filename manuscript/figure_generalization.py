#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from scipy import stats
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm
import warnings
import matplotlib as mpl
import plot_utils

warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')

#%%

class PlotData:
    # loading data is too slow to comfortably iterate on the plot
    # hence I store everything in this object and only load once
    
    def __init__(self):
        self.load_ot1_test_data()
        self.load_naive_test_data()
        self.load_auc_data()
        self.load_ergo_data()
    
    def load_ot1_test_data(self):

        # tcr distances
        ddf = pd.read_csv(
            #'../data/distances_activated_tcrs.csv'  # cdrdist
            '../data/tcrdist_all_tcrs.csv'  # tcrdist
        ).set_index('Unnamed: 0')
                
        # copy OT1 distances to new names
        ddf['OTI_PH'] = ddf['OT1']
        ddf['LR_OTI_1'] = ddf['OT1']
        ddf['LR_OTI_2'] = ddf['OT1']
        ddf.loc['OTI_PH'] = ddf.loc['OT1']
        ddf.loc['LR_OTI_1'] = ddf.loc['OT1']
        ddf.loc['LR_OTI_2'] = ddf.loc['OT1']
        
        # normalize all values to [0, 1]
        ddf /= 1e-9 + ddf.values.max()
        
        ddf = ddf.reset_index().melt(
            'Unnamed: 0'
        ).rename(columns={
            'Unnamed: 0': 'train_tcr',
            'variable': 'test_tcr',
            'value': 'tcrdist',
        }).applymap(
            lambda s: s.upper() if isinstance(s, str) else s
        ).replace({
            'ED161': 'ED16-1',
            'ED1630': 'ED16-30'
        })
        
        fname = 'results/cross-performance.csv.gz'
        pdf = pd.read_csv(fname)
        
        adf = pdf[
            pdf['train_tcr'] == pdf['test_tcr']
        ].sort_values(['mut_pos', 'mut_ami'])
        
        cdf = pd.DataFrame([{
            'train_tcr': t1,
            'test_tcr': t2,
            'spearman': stats.spearmanr(
                adf[adf['test_tcr'] == t1]['activation'].values,
                adf[adf['test_tcr'] == t2]['activation'].values
            )[0]
        } for t1, t2 in itertools.product(
            pdf['train_tcr'].unique(), pdf['train_tcr'].unique()
        )])
        
        pairs = pdf.groupby(['train_tcr', 'test_tcr']).apply(lambda q: pd.Series({
            'auc': metrics.roc_auc_score(q['is_activated'], q['pred']),
            'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
        })).reset_index().merge(
            ddf, on=['train_tcr', 'test_tcr'],
        ).merge(cdf, on=['train_tcr', 'test_tcr'])
        
        dd = pairs.query('test_tcr=="OTI_PH"')[[
            'train_tcr', 'auc', 'tcrdist'
        ]].rename(columns={
            'train_tcr': 'Other TCR',
            'auc': 'AUC (Test on OTI_PH)'
        }).merge(
            pairs.query('train_tcr=="OTI_PH"')[[
                'test_tcr', 'auc',
            ]].rename(columns={
                'test_tcr': 'Other TCR',
                'auc': 'AUC (Train on OTI_PH)'
            }),
            on='Other TCR'
        ).sort_values('AUC (Test on OTI_PH)')
        
        dd['TCR-closeness'] = 1 - dd['tcrdist']
        
        self.ot1_test_data = dd[~dd['Other TCR'].str.contains('OTI')]
        
        
    def load_naive_test_data(self):
        fname = 'results/cross-performance-educated-vs-naive.csv.gz'
        pdf = pd.read_csv(fname)
        
        pp = pdf.query('~is_educated').groupby(
            'tcr', as_index=False
        ).apply(lambda q: pd.Series({
            'auc': metrics.roc_auc_score(q['is_activated'], q['pred']),
            'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
        }))
            
        self.naive_test_data = pp
        
    def load_auc_data(self):
        fname = 'results/tcr_stratified_classification_performance.csv.gz'
        pdf = pd.read_csv(fname)
        
        pdf = pdf.query('mut_pos >= 0')
        pdf = pdf.groupby([
            'normalization', 'tcr'
        ]).filter(lambda g: 0 < g['is_activated'].sum() < len(g) - 1)

        pdf['is_educated'] = pdf['tcr'].str.startswith('ED')
        pdf['Repertoire'] = np.where(pdf['is_educated'], 'Educated', 'Naive')

        self.auc_data = pdf
        
    def load_ergo_data(self):
        def get_auc_per_tcr(data, tcr, y_true, ergo=True):
            data_tmp = data[data['tcr']==tcr]
            col_pred = 'pred'
            if ergo:
                col_pred = 'Score'
            y_pred = data_tmp[col_pred]
            auc_score = metrics.roc_auc_score(y_true, y_pred)
            return auc_score  

        path_res = f'results/ergo2_vdjdb.csv'
        prediction_ergo = pd.read_csv(path_res, index_col=0)
        prediction_ergo = prediction_ergo[prediction_ergo['Peptide']!='SIINFEKL']
        
        path_our = 'results/tcr_stratified_classification_performance.csv.gz'
        prediction_ours = pd.read_csv(path_our, compression='gzip')
        prediction_ours = prediction_ours[prediction_ours['normalization']=='AS']
        prediction_ours = prediction_ours[prediction_ours['threshold']==46.9]
        prediction_ours = prediction_ours[prediction_ours['tcr'].isin(prediction_ergo['tcr'].unique())]
        prediction_ours = prediction_ours[prediction_ours['reduced_features']]
        
        performance = []
        for tcr in prediction_ergo['tcr'].unique():
            y_true = prediction_ours[prediction_ours['tcr']==tcr]['is_activated']
            auc_ergo = get_auc_per_tcr(prediction_ergo, tcr, y_true)
            auc_ours = get_auc_per_tcr(prediction_ours, tcr, y_true, ergo=False)
            performance.append([tcr, auc_ergo, auc_ours])
        performance = pd.DataFrame(performance, columns=['tcr', 'ergo', 'ours'])
        performance = performance[~performance['tcr'].isin(['LR_OTI_1', 'LR_OTI_2'])]
        performance = performance.set_index('tcr')
        
        self.performance = performance


plot_data = PlotData()

#%%

class Plotter:
    # lazy - since I do not want to pass one hundred variables around
    # I will just set everything in self
    
    def __init__(self, plot_data):
        self.textwidth = 6.7261  # in
        self.linewidth = 3.2385  # in
        self.dpi = 350
        self.figfmt = 'tif'
        
        self.plot_data = plot_data
    
    def plot_ot1_auc(self, fig, ax):
        g = sns.regplot(
            data=self.plot_data.ot1_test_data,
            x='AUC (Test on OTI_PH)',
            y='tcrdist',
            ax=ax
        )
        g.set(xlim=(0.3, 1), ylim=(0.4, 1))
        
        ax.grid(False)
        sns.despine(ax=ax)

    def plot_naive_test(self, fig, ax):
        sns.stripplot(
            data=self.plot_data.naive_test_data.sort_values('auc', ascending=False),
            x='tcr', y='auc',
            ax=ax,
        )
        
        ax.tick_params(axis='x', rotation=90)    
        ax.grid(False)
        sns.despine(ax=ax)

    def plot_aucs(self, fig, ax):
        naive_colors = sns.color_palette(
            'Oranges', n_colors=2 * len(
                self.plot_data.auc_data.query('~is_educated')['tcr'].unique()
            ) + 4
        )
        naive_idx = 1
        
        educated_colors = sns.color_palette(
            'Blues', n_colors=2 * len(
                self.plot_data.auc_data.query('is_educated')['tcr'].unique()
            ) + 4
        )
        educated_idx = 1
        
        groups = self.plot_data.auc_data.query(
            'reduced_features & normalization == "AS" & threshold == 46.9'
        ).groupby(['reduced_features', 'tcr'])
        for i, ((fs, tcr), g) in enumerate(groups):
            fpr, tpr, _ = metrics.roc_curve(g['is_activated'], g['pred'])
            auc = metrics.roc_auc_score(g['is_activated'], g['pred'])
            #tpr, fpr, _ = metrics.precision_recall_curve(g['is_activated'], g['pred'])
            #auc = metrics.average_precision_score(g['is_activated'], g['pred'])
        
            if tcr.startswith('ED'):
                educated_idx += 1
                c = educated_colors[educated_idx]
            elif tcr == 'OTI_PH':
                c = 'C2'
            else:
                naive_idx += 1
                c = naive_colors[naive_idx]
        
            ax.plot(fpr, tpr, c=c)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        
        sns.despine(ax=ax)
        ax.grid(False)
        
    def plot_ergo_data(self, fig, ax):
        educated_colors = sns.color_palette(
            'Oranges', n_colors=16 + 4
        )
        educated_idx = 1
        
        naive_colors = sns.color_palette(
            'Blues', n_colors=11 + 4
        )
        naive_idx = 1
        
        ot1_colors = sns.color_palette(
            'Greens', n_colors=2
        )
        ot1_idx = 1
        
        for row in self.plot_data.performance.iterrows():
            if row[0] == 'OTI_PH':
                c = ot1_colors[ot1_idx]
                ot1_idx += 1
            elif row[0].startswith('ED'):
                c = educated_colors[educated_idx]
                educated_idx += 1
            else:
                c = naive_colors[naive_idx]
                naive_idx += 1
            ax.plot(row[1]['ergo'], row[1]['ours'], c=c, marker='o', markersize=4)
        
        ax.grid(False)
        sns.despine(ax=ax)
        ax.set_xlabel('AUC Ergo2')
        ax.set_ylabel('AUC Ours')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        
        ax.plot((0, 1), (0, 1), 'r--')
        
    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2,
            figsize=(self.textwidth, self.textwidth / 1.5),
            dpi=self.dpi,
            gridspec_kw={
                'height_ratios': [3.25, 2],
            },
        )
        
        self.plot_aucs(fig, ax1)
        self.plot_ergo_data(fig, ax2)
        self.plot_ot1_auc(fig, ax3)
        self.plot_naive_test(fig, ax4)
        
        fig.text(0.03, 1, '(a)', size='large', weight='bold')
        fig.text(0.97, 1, '(b)', size='large', weight='bold')
        fig.text(0.03, 0.47, '(c)', size='large', weight='bold')
        fig.text(0.97, 0.44, '(d)', size='large', weight='bold')
        
        fig.legend([
            mpl.lines.Line2D([], [], c='C0'),
            mpl.lines.Line2D([], [], c='C1'),
            mpl.lines.Line2D([], [], c='C2'),
        ], [
            'Educated', 'Naive', 'OT1'
        ], ncol=3,
            bbox_to_anchor=(0.5, 1.05), loc="upper center"
        )
        
        fig.tight_layout()
        fig.savefig('figures/manuscript_generalization.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        

plotter = Plotter(plot_data)
plotter.plot()

