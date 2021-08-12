#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from scipy import stats
import json

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
        self.load_perf_data()
        self.load_active_learning_data()
        
    def load_perf_data(self):
        def compute_metrics(g):
            return pd.Series({
                'MAE': g['abserr'].mean(),
                'R2': metrics.r2_score(g['act'], g['pred']),
                'Pearson': g['act'].corr(g['pred'], method='pearson'),
                'Spearman': g['act'].corr(g['pred'], method='spearman'),
                'AUC': (
                    metrics.roc_auc_score(g['is_activated'], g['pred_prob'])
                    if np.isfinite(g['pred_prob']).all() and 0 < g['is_activated'].mean() < 1
                    else np.nan
                ),
                'APS': (
                    metrics.average_precision_score(g['is_activated'], g['pred_prob'])
                    if np.isfinite(g['pred_prob']).all()
                    else np.nan
                ),
            })
        
        fname = 'results/tcr_specific_data_size.csv.gz'
        ppdf = pd.read_csv(fname)
        
        # compute metric for each validation fold separately
        mdf = pd.concat([
            # except for lmo CV where each validation fold contained a single sample
            # in that case we just compute a global average for each tcr
            ppdf.query('features=="lmo"') \
                .groupby(['features', 'normalization', 'tcr']) \
                .apply(compute_metrics).reset_index(),
            ppdf.query('features!="lmo"') \
                .groupby(['features', 'normalization', 'tcr', 'fold']) \
                .apply(compute_metrics).reset_index(),
        ])
        
        mdf['features'] = mdf['features'].str.upper()
        
        lmdf = mdf.melt(
            id_vars=['tcr', 'features', 'normalization', 'fold'],
            value_vars=['R2', 'Pearson', 'Spearman', 'MAE', 'APS', 'AUC'],
            var_name='Metric'
        ).rename(
            columns={'features': 'Split', 'value': 'Value'}
        )
        
        lmdf['Repertoire'] = np.where(lmdf['tcr'].str.startswith('ED'), 'Educated', 'Naive')
        ppdf['Repertoire'] = np.where(ppdf['tcr'].str.startswith('ED'), 'Educated', 'Naive')
        
        # fname = 'results/tcr_specific_classification_performance.csv.gz'
        # pdf = pd.read_csv(fname)
        
        # pp = pdf.groupby(
        #     ['reduced_features', 'normalization', 'tcr', 'threshold'], as_index=False
        # ).apply(lambda q: pd.Series({
        #     'auc': metrics.roc_auc_score(q['is_activated'], q['pred']),
        #     'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
        # })).melt(['reduced_features', 'normalization', 'tcr', 'threshold']).pivot_table(
        #     'value', ['tcr', 'threshold', 'normalization', 'reduced_features'], 'variable'
        # ).reset_index()
        
        # pp.loc[:, 'reduced_features'] = np.where(pp['reduced_features'], 'redux', 'full')
        # pp.loc[:, 'Repertoire'] = np.where(pp['tcr'].str.startswith('ED'),
        #                                     'Educated', 'Naive')
        
        # aucs = pp.query('reduced_features == "redux" & normalization == "AS" & threshold == 46.9')
        # aucs.loc[:, 'repertoire'] = np.where(aucs['tcr'].str.startswith('ED'),
        #                                      'educated', 'naive')
    
        # return ppdf, pdf, aucs, pp, lmdf
        #return ppdf, None, None, None, lmdf
        
        self.ppdf = ppdf
        self.lmdf = lmdf
    
    def load_active_learning_data(self): 
        def correlation(method):
            def corr_method(y_true, y_pred):
                df = pd.DataFrame()
                df['y_true'] = y_true
                df['y_pred'] = y_pred
                corr = df['y_true'].corr(df['y_pred'], method=method)
                return corr
            return corr_method
        
        
        metrics_cls = [
            {
                'auc': metrics.roc_auc_score,
                'aps': metrics.average_precision_score,
            },
            {
                'F1': metrics.f1_score,
                'Accuracy': metrics.accuracy_score,
                'Precision': metrics.precision_score,
                'Recall': metrics.recall_score,
            }
        ]
        
        
        metrics_reg = {
            'MAE': metrics.mean_absolute_error,
            'R2': metrics.r2_score,
            'Pearson': correlation(method='pearson'),
            'Spearman': correlation(method='spearman'),
        }
        
        path_out = f'results/al/TEST_act_avg_8.json'
        with open(path_out) as f:
            results_act_avg = json.load(f)
    
        path_out = f'results/al/TEST_rdm_8.json'
        with open(path_out) as f:
            results_rdm = json.load(f) 
        
        def read_greedy_bound(n):
            path_in = f'results/al/test_set_within_{n}/'
            res_files = [path_in + f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f))]
        
            results_upper = {}
            for name in list(metrics_reg.keys()) + list(metrics_cls[0].keys()) + list(metrics_cls[1].keys()):
                results_upper[name] = []
        
            for path_file in res_files:
                with open(path_file) as f:
                    res_tmp = json.load(f)
                for mtc, vals in res_tmp.items():
                    results_upper[mtc] += vals
            return results_upper
        results_upper = read_greedy_bound(8)
        
        self.summary_8 = {
            'active_avg': results_act_avg,
            'random': results_rdm,
            'upper bound': results_upper,
        }


plot_data = PlotData()

#%% statistics for paper

pd.set_option('display.expand_frame_repr', False)
print('\n\n---  summary statistics\n',
      plot_data.lmdf.query('Split=="LMO"') \
          .groupby(['Metric', 'Repertoire'])['Value'].describe())
print('\n\n---  worst two educated\n',
      plot_data.lmdf.query('Split=="LMO"&Metric=="APS"&Repertoire=="Educated"') \
          .sort_values('Value').head(2))
print('\n\n---  worst two naive\n',
      plot_data.lmdf.query('Split=="LMO"&Metric=="APS"&Repertoire=="Naive"') \
          .sort_values('Value').head(2))
print('\n\n---  ot1 performance\n',
      plot_data.lmdf.query('Split=="LMO"&Metric=="APS"&tcr=="OTI_PH"'))
print('\n\n---  spearman by position averaging spearman of individual tcrs\n',
      plot_data.ppdf.query('features=="lpo"').groupby(['fold', 'tcr']).apply(
          lambda g: pd.Series(stats.spearmanr(g['act'], g['pred']), index=['rho', 'p'])
          ).reset_index().groupby('fold').mean().describe()
)
print('\n\n---  spearman by amino acid averaging spearman of individual tcrs\n',
      plot_data.ppdf.query('features=="lao"').groupby(['fold', 'tcr']).apply(
          lambda g: pd.Series(stats.spearmanr(g['act'], g['pred']), index=['rho', 'p'])
          ).reset_index().groupby('fold').mean().describe()
)

# t-test to find significance of split performance reduction
print('\n\n---  data size hypothesis testing')
m = 'Spearman'
plot_data.lmdf.query(f'Metric == "{m}"').groupby(['Split', 'tcr'])['Value'].mean().reset_index()
lmo_aps = plot_data.lmdf.query(f'Split=="LMO" & Metric == "{m}"').groupby('tcr')['Value'].mean().dropna()
for split in plot_data.lmdf['Split'].unique():
    if split == 'LMO':
        continue
    
    split_aps = plot_data.lmdf.query(f'Split=="{split}" & Metric == "{m}"').groupby('tcr')['Value'].mean().dropna()
    assert np.all(lmo_aps.index == split_aps.index)
    tr = stats.wilcoxon(lmo_aps, split_aps, alternative='greater')
    md = split_aps.mean() - lmo_aps.mean()
    sd = np.sqrt((split_aps.var() + lmo_aps.var()) / 2)
    
    print(f'split {split} vs. LMO - mean difference {md:.4f} - effect size {md / sd:.4f} - W: {tr.statistic:.4f} , p: {tr.pvalue:.3e}')


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
        
    def plot_prc(self, fig, ax):
        
        #worst_edu, worst_nai = aucs.set_index('tcr').groupby('repertoire')['aps'].idxmin()
        #best_edu, best_nai = aucs.set_index('tcr').groupby('repertoire')['aps'].idxmax()
        
        #groups = pdf.query('reduced_features & normalization == "AS" & threshold == 46.9').groupby('tcr')
        groups = self.plot_data.ppdf.query('features == "lmo"').groupby('tcr')
        for i, (tcr, g) in enumerate(groups):
            fpr, tpr, _ = metrics.roc_curve(g['is_activated'], g['pred'])
            pre, rec, _ = metrics.precision_recall_curve(g['is_activated'], g['pred'])
        
            auc = metrics.roc_auc_score(g['is_activated'], g['pred'])
            aps = metrics.average_precision_score(g['is_activated'], g['pred'])
        
            # # this is to highlight best and worst for naive / educated repertoire
            # try:
            #     idx = (best_edu, worst_edu, best_nai, worst_nai).index(tcr)
            #     kwargs={
            #         'c': f'C{1 - idx // 2}',
            #         'label': f'{tcr} ({aps:.3f})',
            #         'linestyle': '--' if idx % 2 else '-',
            #     }
            # except ValueError:
            #     kwargs = {
            #         #'c': 'gray',
            #         'c': 'C2' if tcr == 'OTI_PH' else 'C1' if 'ED' in tcr else 'C0',
            #         'alpha': 0.3
            #     }
        
            kwargs = {
                'c': 'C2' if tcr == 'OTI_PH' else 'C1' if 'ED' in tcr else 'C0',
                'alpha': 0.3 if tcr != 'OTI_PH' else 1.0,
                'label': 'OTI_PH' if tcr == 'OTI_PH' else 'Educated' if 'ED' in tcr else 'Naive',
            }
        
            #ax.plot(fpr, tpr, **kwargs)
            ax.plot(rec, pre, **kwargs)
    
        # remove duplicate labels from legend
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), ncol=2, bbox_to_anchor=(0.5, 1.2), loc="upper center")
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(False)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
    
    def plot_aps_boxplot(self, fig, ax2):
        #apss = pp.query('normalization == "AS" & threshold == 46.9 & reduced_features == "redux"')
        
        bp = self.plot_data.lmdf.query('Split == "LMO" & (Metric == "Spearman" | Metric == "APS")')
        sns.boxplot(
            data=bp,
            y='Value', hue='Repertoire', ax=ax2, fliersize=1,
            x='Metric', hue_order=['Naive', 'Educated'],
            color='#ffffffff'
        )
        
        sns.swarmplot(
            data=bp, y='Value', hue='Repertoire', ax=ax2,
            x='Metric', hue_order=['Naive', 'Educated'],
            s=2, dodge=True
        )
        
        # annotate OTI
        ax2.plot([-0.25], [bp.query('tcr=="OTI_PH" & Metric == "Spearman"').iloc[0]['Value']],
                 'C2o', label='OTI_PH')
        ax2.plot([0.75], [bp.query('tcr=="OTI_PH" & Metric == "APS"').iloc[0]['Value']], 'C2o')
        
        #ax2.tick_params(axis='x', rotation=-10)
        ax2.set_ylabel('')
        ax2.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center")
        ax2.grid(False)
        
    def plot_data_size(self, fig, ax3):
        # right boxplot
        fig.tight_layout()
        sns.despine()
        
        data = self.plot_data.lmdf.query('Metric=="Spearman"').groupby([
            'Repertoire', 'Split', 'tcr'
        ])['Value'].agg('mean').reset_index()
        
        order = ['LMO', 'L10O', 'L25O', 'L50O', 'LAO', 'L75O', 'L90O', 'L95O', 'LPO']
        # sns.boxplot(
        #     #data=lmdf.query('Metric=="Spearman"'),
        #     data=data,
        #     x='Split', y='Value', hue='Repertoire',
        #     order=order, hue_order=['Naive', 'Educated'],
        #     ax=ax3, fliersize=1, color='#ffffffff'
        # )
        
        sns.stripplot(
            data=data,
            x='Split', y='Value', hue='Repertoire',
            order=order, hue_order=['Naive', 'Educated'],
            ax=ax3, s=2, dodge=True, #color='k',
        )
        
        sns.pointplot(
            data=data,
            x='Split', y='Value', hue='Repertoire',
            order=order, hue_order=['Naive', 'Educated'],
            ax=ax3, s=2, dodge=True, ci=None,
        )
        
        oti_values = self.plot_data.lmdf.query(
            'Metric == "Spearman" & tcr == "OTI_PH"'
        ).groupby('Split')['Value'].mean().loc[order]
        ax3.plot([x - 0.25 for x in range(len(order))], oti_values.values, 'C2o',
                 label='OTI_PH')
        
        # # statistical significance for APS metric
        # # NB: hardcoded based on the p-values printed when you run the script
        # ax3.plot([0, 0, 5, 5], [1.15, 1.175, 1.175, 1.15], lw=1, c='k')
        # ax3.text(2.5, 1.175, "p = 2.7e-5 ***", ha='center', va='bottom')
        # ax3.plot([0, 0, 3, 3], [1.05, 1.075, 1.075, 1.05], lw=1, c='k')
        # plt.text(1.5, 1.075, "p = 0.36", ha='center', va='bottom')
        
        #ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #ax3.tick_params(axis='x', rotation=90)
        ax3.set_ylabel('Spearman ')
        ax3.grid(False)
        ax3.legend(ncol=3, bbox_to_anchor=(0.5, 1.2), loc="upper center", title='Repertoire')
    
    def plot_active_learning(self, fig, ax1, ax2):
        N = 8
        M = 10
        
        palette = {'active_avg': 'tab:red', 'upper bound': 'tab:olive', 'random': 'tab:cyan'}
        
        for name, ax in zip(['auc', 'Spearman'], [ax1, ax2]):
            dfs_results = []
            for method in self.plot_data.summary_8.keys():
                df = pd.DataFrame(self.plot_data.summary_8[method][name])
                df.columns = ['tcr', 'is_educated', 'iteration', name]
                df['method'] = method
                dfs_results.append(df)
                
            df_joint = pd.concat(dfs_results)
            plot = sns.lineplot(
                data=df_joint, x='iteration', y=name, hue='method',
                palette=palette, ax=ax
            )
            sns.despine(bottom=False, left=False)
            
            plot.set(xlabel='Amount Samples')
            if len(name)<=3:
                plot.set(ylabel=name.upper())
                
            ax.legend(title='Sampling Method', labels=['Active', 'Random', 'Upper Bound'])
            x_ticks = list(range(0, M))
            x_labels = ['9'] + [str(9+(i+1)*N) for i in range(M-1)]
            plot.set_xticks(x_ticks)
            plot.set_xticklabels(x_labels)
            ax1.grid(False)
            ax2.grid(False)
    
    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)
        
        fig = plt.figure(
            figsize=(self.textwidth, self.textwidth / 1.5),
            dpi=self.dpi,
        )
        
        gridspecs = {}
        axes = {}
        
        gridspecs["gs_12345"] = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=2,
            ncols=1,
            height_ratios=[1, 1],
            width_ratios=[6],
        )
        
        gridspecs["gs_123"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345"][0],
            nrows=1,
            ncols=3,
            height_ratios=[2],
            width_ratios=[2, 1, 3],
        )
        axes["ax_1"] = fig.add_subplot(gridspecs["gs_123"][0])
        axes["ax_2"] = fig.add_subplot(gridspecs["gs_123"][1])
        axes["ax_3"] = fig.add_subplot(gridspecs["gs_123"][2])
        
        gridspecs["gs_45"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345"][1],
            nrows=1,
            ncols=2,
            height_ratios=[1],
            width_ratios=[3, 3],
            wspace=0.2,
            hspace=0.5,
        )
        axes["ax_4"] = fig.add_subplot(gridspecs["gs_45"][0])
        axes["ax_5"] = fig.add_subplot(gridspecs["gs_45"][1])
        
        self.plot_prc(fig, axes['ax_1'])
        self.plot_aps_boxplot(fig, axes['ax_2'])
        self.plot_data_size(fig, axes['ax_3'])
        self.plot_active_learning(fig, axes['ax_4'], axes['ax_5'])
        
        fig.text(0.05, 0.95, '(a)', size='large', weight='bold')
        fig.text(0.33, 0.95, '(b)', size='large', weight='bold')
        fig.text(0.95, 0.95, '(c)', size='large', weight='bold')
        fig.text(0.04, 0.49, '(d)', size='large', weight='bold')
        fig.text(0.95, 0.49, '(e)', size='large', weight='bold')
        
        axes['ax_1'].get_legend().remove()
        axes['ax_3'].get_legend().remove()
        axes['ax_2'].legend([
            mpl.patches.Patch(facecolor='C0'),
            mpl.patches.Patch(facecolor='C1'),
            mpl.patches.Patch(facecolor='C2'),
        ], [
            'Naive', 'Educated', 'OTI_PH',
        ], ncol=3, bbox_to_anchor=(1, 1.15), loc='upper center')
        
        fig.tight_layout()
            
        fig.savefig('figures/manuscript_within_tcr.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        

Plotter(plot_data).plot()