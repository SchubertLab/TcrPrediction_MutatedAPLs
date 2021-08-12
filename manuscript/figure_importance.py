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
        self.load_data_permutation_importance()
        self.load_position_importance_data()
        
    def load_data_permutation_importance(self):
        fname = 'results/tcr_stratified_permutation_importance.csv.gz'
        pdf = pd.read_csv(fname)
    
        mdf = pdf[pdf['tcr'].isin(
            pdf[pdf['is_activated'] > 0.5]['tcr'].unique()
        )].groupby(['tcr', 'group', 'shuffle']).apply(lambda q: pd.Series({
            'auc': metrics.roc_auc_score(q['is_activated'], q['pred']),
            'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
            #'spearman': stats.spearmanr(q['activation'], q['pred'])[0],
        })).reset_index().drop(columns='shuffle')
        
        ddf = mdf.melt(['tcr', 'group']).merge(
            mdf[
                mdf['group'] == 'all'
            ].drop(columns='group').melt('tcr', value_name='base').drop_duplicates(),
            on=['tcr', 'variable']
        )
        ddf['diff'] = ddf['value'] - ddf['base']
        ddf['rel'] = ddf['value'] / ddf['base'] - 1  # positive = increase
        ddf['item'] = ddf['group'].str.split('_').str[0]
        ddf['is_educated'] = np.where(
            ddf['tcr'].str.startswith('ED'),
            'Educated', 'Naive'
        )
        
        self.permutation_data = ddf[(
            ddf['is_educated'] == "Educated"
        ) & (
            ddf['variable'] == 'auc'
        ) & (
            ddf['group'].str.startswith('pos_')
              | ddf['group'].isin(['cdr3', 'all'])
        )].rename(columns={
            'value': 'AUC', 'group': 'Permutation'
        }).replace({
            'pos_0': 'P1', 'pos_1': 'P2', 'pos_2': 'P3', 'pos_3': 'P4',
            'pos_4': 'P5', 'pos_5': 'P6', 'pos_6': 'P7', 'pos_7': 'P8',
            'cdr3': 'CDR3', 'all': '-'
        })
            
    def load_position_importance_data(self):
        fname = 'results/tcr_stratified_leave_position_out_performance.csv.gz'
        pdf = pd.read_csv(fname)
        
        pp = pdf.groupby(
            ['tcr', 'mut_pos'], as_index=False
        ).apply(lambda q: pd.Series({
            'auc': metrics.roc_auc_score(
                q['is_activated'], q['pred']
            ) if 0 < q['is_activated'].mean() < 1 else np.nan,
            'aps': metrics.average_precision_score(q['is_activated'], q['pred']),
        }))
        
        pp['is_educated'] = pp['tcr'].str.startswith('ED')
        
        self.importance_data = pp


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
        
    def plot_permutation_importance(self, fig, ax):
        g = sns.boxplot(
            data=self.plot_data.permutation_data,
            x='Permutation',
            y='AUC',
            palette='husl',
            zorder=2,
            showmeans=True,
            notch=False,
            meanprops={'mfc': 'k', 'mec': 'k'},
            ax=ax,
        )
        
        g.set(ylim=(0.5, 1), ylabel='AUC on educated repertoire',
              xlabel='Group permuted')
        
        ax.grid(False)
        sns.despine(ax=ax)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(a)')
    
    def plot_position_importance(self, fig, ax):
        g = sns.boxplot(
            data=self.plot_data.importance_data,
            x='mut_pos',    
            y='auc',
            #col='is_educated',
            #hue='is_educated',
            palette='husl',
            zorder=2,
            showmeans=True,
            meanprops={'mfc': 'k', 'mec': 'k'},
            ax=ax,
        )
        
        g.set(
            xticklabels=[f'P{i+1}' for i in range(8)],
            xlabel='Validate on position',
            ylabel='AUC',
        )
        
        ax.grid(False)
        sns.despine(ax=ax)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(b)')
        
    def plot_3d_structures(self, fig, ax1, ax2, ax3):
        
        def do_plot(ax, idx, path):
            ax.imshow(mpl.image.imread(path))
            ax.grid(False)
            sns.despine(ax=ax, left=True, bottom=True)
            plot_utils.dd_axis_title_fixed_position(fig, ax, f'({idx})')
            ax.set_xticks([])
            ax.set_yticks([])
            
        do_plot(ax1, 'c', 'figures/OT1_SIINFEKL.png')
        do_plot(ax2, 'd', 'figures/B11_SIINFEKL.png')
        do_plot(ax3, 'e', 'figures/B14_SIINFEKL.png')

    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)

        fig = plt.figure(figsize=(self.textwidth, self.textwidth / 1.5),
                         dpi=self.dpi)
        gridspecs = {}
        axes = {}
        
        gridspecs["gs_12345"] = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=2,
            ncols=1,
            height_ratios=[1, 1],
            width_ratios=[6],
            #wspace=0.1,
            #hspace=2/3,
        )
        
        gridspecs["gs_12"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345"][0],
            nrows=1,
            ncols=2,
            height_ratios=[2],
            width_ratios=[3, 3],
            #wspace=0.2,
            #hspace=0.25,
        )
        axes["ax_1"] = fig.add_subplot(gridspecs["gs_12"][0])
        axes["ax_2"] = fig.add_subplot(gridspecs["gs_12"][1])
        
        gridspecs["gs_345"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345"][1],
            nrows=1,
            ncols=3,
            height_ratios=[1],
            width_ratios=[2, 2, 2],
            #wspace=0.3,
            #hspace=0.5,
        )
        axes["ax_3"] = fig.add_subplot(gridspecs["gs_345"][0])
        axes["ax_4"] = fig.add_subplot(gridspecs["gs_345"][1])
        axes["ax_5"] = fig.add_subplot(gridspecs["gs_345"][2])
        
        self.plot_permutation_importance(fig, axes['ax_1'])
        self.plot_position_importance(fig, axes['ax_2'])
        self.plot_3d_structures(fig, axes['ax_3'], axes['ax_4'], axes['ax_5'])
        
        fig.tight_layout()
        fig.savefig('figures/manuscript_importance.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        

plotter = Plotter(plot_data)
plotter.plot()

