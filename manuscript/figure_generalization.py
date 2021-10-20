#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
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
        self.load_baseline_data()
        self.load_active_learning_data()

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

        aucs = pdf.query(
                'reduced_features & normalization == "AS" & threshold == 46.9'
        ).groupby([
            'is_educated', 'tcr'
        ], as_index=False).apply(lambda g: pd.Series({
            'auc': metrics.roc_auc_score(g['is_activated'], g['pred'])
        }))
        
        aucs['is_educated'] = np.array(aucs['is_educated'].values, dtype=np.bool)

        self.auc_data = pdf
        self.aucs = aucs
        
    def load_baseline_data(self):
        def get_auc_per_tcr(data, tcr, y_true, col_pred):
            data_tmp = data[data['tcr'] == tcr]
            y_pred = data_tmp[col_pred]
            auc_score = metrics.roc_auc_score(y_true, y_pred)
            return auc_score

        prediction_ergo = self.load_ergo_data()
        prediction_ours = self.load_ours_data()
        prediction_imrex = self.load_imrex_data()
        prediction_titan = self.load_titan_data()

        prediction_ours = prediction_ours[prediction_ours['tcr'].isin(prediction_ergo['tcr'].unique())]

        performance = []
        for tcr in prediction_ergo['tcr'].unique():
            if tcr in ['LR_OTI_1', 'LR_OTI_2']:
                continue
            y_true = prediction_ours[prediction_ours['tcr'] == tcr]['is_activated']
            auc_ergo = get_auc_per_tcr(prediction_ergo, tcr, y_true, col_pred='Score')
            auc_imrex = get_auc_per_tcr(prediction_imrex, tcr, y_true, col_pred='prediction_score')
            auc_titan = get_auc_per_tcr(prediction_titan, tcr, y_true, col_pred='Score')
            auc_ours = get_auc_per_tcr(prediction_ours, tcr, y_true, col_pred='pred')
            performance.append([tcr, auc_ergo, auc_imrex, auc_titan, auc_ours])
        performance = pd.DataFrame(performance, columns=['tcr', 'ergo', 'imrex', 'titan', 'ours'])
        performance = performance[~performance['tcr'].isin(['LR_OTI_1', 'LR_OTI_2'])]
        performance = performance.set_index('tcr')
        order = list(performance.index)
        order.remove('OTI_PH')
        order = ['OTI_PH'] + order
        performance = performance.reindex(order)
        self.performance = performance

    def load_ours_data(self):
        path_our = 'results/tcr_stratified_classification_performance.csv.gz'
        prediction_ours = pd.read_csv(path_our, compression='gzip')
        prediction_ours = prediction_ours[prediction_ours['normalization'] == 'AS']
        prediction_ours = prediction_ours[prediction_ours['threshold'] == 46.9]
        return prediction_ours

    def load_ergo_data(self):
        path_res = f'results/ergo2_vdjdb.csv'
        prediction_ergo = pd.read_csv(path_res, index_col=0)
        prediction_ergo = prediction_ergo[prediction_ergo['Peptide'] != 'SIINFEKL']
        return prediction_ergo

    def load_imrex_data(self):
        path_imrex = f'results/MG_imrex.csv'
        prediction_imrex = pd.read_csv(path_imrex, index_col=0)
        prediction_imrex = prediction_imrex[prediction_imrex['antigen.epitope'] != 'SIINFEKL']
        return prediction_imrex

    def load_titan_data(self):
        path_res = f'results/output_titan_prediction.npy'
        prediction_titan = pd.read_csv(f'results/ergo2_vdjdb.csv',
                                       index_col=0)  # read ergo for the remaining information
        prediction_titan['Score'] = np.load(path_res)[0]
        prediction_titan = prediction_titan[prediction_titan['Peptide'] != 'SIINFEKL']
        return prediction_titan

    def load_active_learning_data(self):
        results_active = self.load_active()
        results_random = self.load_random()
        results_upper = self.load_upper()

        self.summary_active = {
            'active': results_active,
            'random': results_random,
            'upper bound': results_upper,
        }

    def load_active(self):
        path_res = 'results/active_learning/across/crossTCR_FULL_active_8_10.json'
        with open(path_res) as f:
            results = json.load(f)
        return results

    def load_random(self):
        path_res = 'results/active_learning/across/crossTCR_FULL_random_8_10.json'
        with open(path_res) as f:
            results = json.load(f)
        return results

    def load_upper(self):
        path_in = f'results/active_learning/across/greedy/'
        res_files = [path_in + f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f))]

        results_upper = {}

        for path_file in res_files:
            with open(path_file) as f:
                res_tmp = json.load(f)
            for mtc, vals in res_tmp.items():
                if mtc not in results_upper:
                    results_upper[mtc] = []
                results_upper[mtc] += vals
        return results_upper


plot_data = PlotData()

#%% numbers for manuscript
    
print('\nauc summary\n',
    plot_data.aucs.groupby('is_educated').describe().T
)

print('\nbest edu\n', plot_data.aucs.query('is_educated').sort_values('auc').tail(3))
print('\nworst edu\n', plot_data.aucs.query('is_educated').sort_values('auc').head(3))
print('\nbest naive\n', plot_data.aucs.query('~is_educated').sort_values('auc').tail(3))
print('\nworst naive\n', plot_data.aucs.query('~is_educated').sort_values('auc').head(3))
print('\nbelow random\n', plot_data.aucs.query('auc < 0.5').sort_values('auc'))
print('\ncorrelation between test oti and tcrdist\n', stats.spearmanr(
    plot_data.ot1_test_data[['AUC (Test on OTI_PH)', 'tcrdist']]
))
print('\ncomparing auc of leave-tcr-out and leave-naive-out\n', stats.ttest_rel(*pd.merge(
     plot_data.naive_test_data,
     plot_data.aucs,
     left_on='tcr', right_on='tcr',
)[['auc_x', 'auc_y']].values.T))


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
        self.plot_data.ot1_test_data['repertoire'] = np.where(
            self.plot_data.ot1_test_data['Other TCR'].str.startswith('ED'),
            'educated', 'naive'
        )
        
        g = sns.regplot(
            data=self.plot_data.ot1_test_data.rename(columns={
                'AUC (Test on OTI_PH)': 'AUC (Test on OTI)',
                'tcrdist': 'TCRdist',
            }),
            x='AUC (Test on OTI)',
            y='TCRdist',
            color=sns.color_palette("Greens")[3],
            ax=ax,
        )
        g.set(xlim=(0.3, 1), ylim=(0.4, 1))
        
        ax.grid(False)
        sns.despine(ax=ax)

    def plot_naive_test(self, fig, ax):
        dd = pd.merge(
             plot_data.naive_test_data,
             plot_data.aucs,
             left_on='tcr', right_on='tcr',
        ).rename(columns={
            'auc_x': 'Leave-naive-out',
            'auc_y': 'Leave-TCR-out',
            'tcr': 'TCR',
        }).melt(
            'TCR',
            ['Leave-naive-out', 'Leave-TCR-out'],
            var_name='Validation',
            value_name='AUC'
        ).replace({
            'TCR': {'OTI_PH': 'OTI'}
        }).sort_values('AUC', ascending=False)
        
        sns.barplot(
            data=dd,
            x='TCR', y='AUC', hue='Validation',
            palette='Blues',
            ax=ax, alpha=0.5,
        )
        
        # re-draw OTI bars in green
        g0, g1, g2, g3 = sns.color_palette("Greens")[:4]
        ax.bar(
            [1.8],
            dd.query('TCR=="OTI" & Validation == "Leave-naive-out"')['AUC'].values,
            width=0.38, color=g1, edgecolor=g0, linewidth=0.4,
        )
        ax.bar(
            [2.2],
            dd.query('TCR=="OTI" & Validation == "Leave-TCR-out"')['AUC'].values,
            width=0.38, color=g3, edgecolor=g1, linewidth=0.4,
        )
        
        ax.legend(title='Validation Strategy')
        #ax.tick_params(axis='x', rotation=90)    
        ax.grid(False)
        sns.despine(ax=ax)

    def plot_aucs(self, fig, ax):
        naive_colors = sns.color_palette(
            'Blues', n_colors=2 * len(
                self.plot_data.auc_data.query('~is_educated')['tcr'].unique()
            ) + 4
        )
        naive_idx = 1
        
        educated_colors = sns.color_palette(
            'Oranges', n_colors=2 * len(
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
                c = sns.color_palette("Greens")[3]
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
        
    def plot_baseline_data(self, fig, ax):
        educated_colors = sns.color_palette(
            'Oranges', n_colors=len(self.plot_data.performance.index) + 4
        )

        naive_colors = sns.color_palette(
            'Blues', n_colors=len(self.plot_data.performance.index) + 4
        )

        color_dict = {}

        educated_idx = 1
        naive_idx = 1
        for tcr in self.plot_data.performance.index:
            if tcr.startswith('ED'):
                educated_idx += 1
                c = educated_colors[educated_idx]
            elif tcr == 'OTI_PH':
                c = 'C2'
            else:
                naive_idx += 1
                c = naive_colors[naive_idx]
            color_dict[tcr] = c

        performance_cat = self.plot_data.performance
        performance_cat['tcr'] = performance_cat.index
        performance_cat = performance_cat.melt(['tcr'], value_name='AUC', var_name='Model')
        performance_cat['color'] = [color_dict[tcr] for tcr in performance_cat['tcr']]

        plot_cat = sns.swarmplot(data=performance_cat, x='Model', y='AUC', hue='tcr',
                               order=['titan', 'imrex', 'ergo', 'ours'],
                               palette=performance_cat['color'], ax=ax)
        plot_cat.set(ylabel='AUC')
        plot_cat.set_xticklabels(['TITAN', 'ImRex', 'ERGO2', '%OURs%'])

        ax.get_legend().remove()
        plot_cat.set(xlabel=None)
        sns.despine(ax=ax)
        ax.grid(False)
    
    def get_axes(self, fig):
        gridspecs = {}
        axes = {}

        gridspecs["gs_123456"] = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=2,
            ncols=1,
            height_ratios=[3, 1.25]
        )

        gridspecs["gs_1234"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_123456"][0],
            nrows=1,
            ncols=2,
            height_ratios=[3],
            width_ratios=[2, 2],
            #wspace=0.3,
            #hspace=0.16666666666666666,
        )
        
        gridspecs["gs_13"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_1234"][0],
            nrows=2,
            ncols=1,
            height_ratios=[2, 1],
            width_ratios=[2],
            #wspace=0.3,
            hspace=0.3333333333333333,
        )
        axes["ax_1"] = fig.add_subplot(gridspecs["gs_13"][0])
        axes["ax_3"] = fig.add_subplot(gridspecs["gs_13"][1])
        
        gridspecs["gs_24"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_1234"][1],
            nrows=2,
            ncols=1,
            height_ratios=[1, 2],
            width_ratios=[2],
            #wspace=0.3,
            hspace=0.3333333333333333,
        )
        axes["ax_2"] = fig.add_subplot(gridspecs["gs_24"][0])
        axes["ax_4"] = fig.add_subplot(gridspecs["gs_24"][1])

        gridspecs["gs_56"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_123456"][1],
            nrows=1,
            ncols=2,
            height_ratios=[1],
            width_ratios=[1, 1]
        )
        axes["ax_5"] = fig.add_subplot(gridspecs["gs_56"][0])
        axes["ax_6"] = fig.add_subplot(gridspecs["gs_56"][1])

        return axes

    def plot_active_learning(self, fig, ax1, ax2):
        cm = plt.get_cmap('tab10')
        palette = {
            'active': plot_utils.interpolate_transparency(cm(3), 0.6),
            'upper bound': plot_utils.interpolate_transparency(cm(4), 0.6),
            'random': plot_utils.interpolate_transparency(cm(5), 0.6),
        }

        for name, ax in zip(['auc', 'Spearman'], [ax1, ax2]):
            dfs_results = []
            for method in self.plot_data.summary_active.keys():
                df = pd.DataFrame(self.plot_data.summary_active[method][name])
                df.columns = ['tcr', 'iteration', name]
                df['method'] = method
                dfs_results.append(df)

            df_joint = pd.concat(dfs_results)
            # plot = sns.lineplot(
            #     data=df_joint, x='iteration', y=name, hue='method',
            #     palette=palette, ax=ax
            # )
            g = sns.pointplot(
                data=df_joint, x='iteration', y=name, hue='method',
                palette=palette, ax=ax, ci=None, dodge=0.5,
                markers='.'
            )
            plt.setp(g.get_lines(), linewidth=1.5)

            sns.stripplot(
                data=df_joint, x='iteration', y=name, hue='method',
                palette=palette, ax=ax, s=1, dodge=True,
            )

            sns.despine(ax=ax, bottom=False, left=False)

            ax.set_xlabel('Amount of Samples')
            if len(name) <= 3:
                ax.set_ylabel(name.upper())

            ax.legend(title='Sampling Method',
                      labels=['Active', 'Random', 'Upper Bound'],
                      loc='lower right')
            x_ticks = list(range(0, 10))
            x_labels = ['9'] + [str(9 + (i + 1) * 8) for i in range(10 - 1)]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax1.grid(False)
            ax2.grid(False)

    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)
        
        fig = plt.figure(figsize=(self.textwidth, self.textwidth / 1.25), #/ 1.5),
                         dpi=self.dpi)
        axes = self.get_axes(fig)
        
        self.plot_aucs(fig, axes['ax_1'])
        self.plot_naive_test(fig, axes['ax_2'])
        self.plot_ot1_auc(fig, axes['ax_3'])
        self.plot_baseline_data(fig, axes['ax_4'])
        self.plot_active_learning(fig, axes['ax_5'], axes['ax_6'])

        fig.text(0.01, 0.99, '(a)', size='large', weight='bold')
        fig.text(0.5, 0.99, '(b)', size='large', weight='bold')
        fig.text(0.01, 0.59, '(c)', size='large', weight='bold')
        fig.text(0.5, 0.74, '(d)', size='large', weight='bold')
        fig.text(0.01, 0.33, '(e)', size='large', weight='bold')
        fig.text(0.5, 0.33, '(f)', size='large', weight='bold')
        # 0.97
        # 0.03

        fig.legend([
            mpl.lines.Line2D([], [], c='C1'),
            mpl.lines.Line2D([], [], c='C0'),
            mpl.lines.Line2D([], [], c=sns.color_palette("Greens")[3]),
        ], [
            'Educated', 'Naive', 'OT1'
        ], ncol=3,
            bbox_to_anchor=(0.5, 1.05), loc="upper center"
        )
        
        fig.tight_layout()
        fig.savefig('figures/manuscript_generalization.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        fig.savefig('figures/manuscript_generalization.png',
                    dpi=self.dpi, bbox_inches='tight')
        

plotter = Plotter(plot_data)
plotter.plot()

