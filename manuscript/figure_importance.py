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
        self.load_3D_distances()

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

    def load_3D_distances(self):
        fname = '../manuscript/results/distances_SIINFEKL.csv'
        distances = pd.read_csv(fname, index_col=0)

        def get_distance_matrices(do_alpha=True):
            col = 'features_beta'
            if do_alpha:
                col = 'features_alpha'
            col = distances[col]

            distance_dict = {}
            for tcr in col.index:
                dist = col[tcr]
                dist = dist.split(',')
                dist = [float(el) for el in dist]
                dist = np.array(dist)
                dist = dist.reshape(8, -1)
                distance_dict[tcr] = dist
            return distance_dict

        distances_alpha = get_distance_matrices()
        distances_beta = get_distance_matrices(do_alpha=False)
        max_value = 0
        min_value = 99
        min_dist_dict = {}
        for tcr in distances_beta:
            dist_alpha = distances_alpha[tcr]
            dist_alpha = np.min(dist_alpha, axis=1)

            dist_beta = distances_beta[tcr]
            dist_beta = np.min(dist_beta, axis=1)

            dist_total = np.stack([dist_alpha, dist_beta])
            dist_total = np.min(dist_total, axis=0, keepdims=True)
            min_dist_dict[tcr] = dist_total
            max_value = max(np.max(dist_total).tolist(), max_value)
            min_value = min(np.min(dist_total).tolist(), min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.distance_dict = min_dist_dict


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

    def plot_avg_dist(self, fig, ax1, ax2):
        dist = self.plot_data.distance_dict
        dist = np.stack(dist.values())
        dist = dist.reshape(-1, 8)
        dist = np.mean(dist, keepdims=True, axis=0)
        dist = dist.T

        y_labels = [f'P{i + 1}' for i in range(8)]
        plot = sns.heatmap(dist, square=True, annot=True,
                           vmin=self.plot_data.min_value, vmax=self.plot_data.max_value, yticklabels=y_labels,
                           cbar=False,
                           ax=ax2)
        plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
        plot.set_ylabel('Distances averaged over TCRs')
        plot.set_xticks([])
        ax2.yaxis.labelpad = 1.5
        ax2.tick_params(left=True)

        norm = mpl.colors.Normalize(self.plot_data.min_value, self.plot_data.max_value)
        cmap = sns.color_palette("rocket", as_cmap=True)
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, label='Distance in Å')
        cb.outline.set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.labelpad = 0
        plot_utils.add_axis_title_fixed_position(fig, ax2, '(c)')

    def plot_3d_structures(self, fig, ax1, ax2, ax3):
        def do_plot(ax, idx, path, tcr, crop_x=None, crop_y=None):
            img = mpl.image.imread(path)
            if crop_x is not None:
                img = img[crop_y[0]:crop_y[1], :, :]
            if crop_y is not None:
                img = img[:, crop_x[0]:crop_x[1], :]
            ax.imshow(img)
            ax.grid(False)
            sns.despine(ax=ax, left=True, bottom=True)
            plot_utils.add_axis_title_fixed_position(fig, ax, f'({idx})')
            ax.set_xticks([])
            ax.set_yticks([])

        do_plot(ax1, 'd', 'figures/OT1_SIINFEKL.png', 'OT1', crop_x=(2540, 4210), crop_y=(150, 2130))  # x x x x
        do_plot(ax2, 'e', 'figures/B11_SIINFEKL.png', 'B11', crop_x=(2640, 4360), crop_y=(310, 2160))  # x x x x
        do_plot(ax3, 'f', 'figures/B14_SIINFEKL.png', 'B14', crop_x=(2470, 4280), crop_y=(310, 2090))  # x y x x

    def plot_3d_distances(self, fig, ax1, ax2, ax3):
        def do_plot(ax, tcr):
            dist = self.plot_data.distance_dict[tcr]
            x_labels = [f'P{i + 1}' for i in range(8)]
            plot = sns.heatmap(dist, square=True,
                               vmin=self.plot_data.min_value, vmax=self.plot_data.max_value,
                               xticklabels=x_labels, cbar=False, ax=ax)
            if tcr == 'OT1':
                tcr = 'OTI'
            plot.set_xlabel(tcr)
            plot.set_yticks([])
            ax.tick_params(bottom=True, width=0.75)
            plot_utils.add_axis_title_fixed_position(fig, ax, None)
        do_plot(ax1, 'OT1')
        do_plot(ax2, 'B11')
        do_plot(ax3, 'B14')

    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)

        fig = plt.figure(figsize=(self.textwidth, self.textwidth / 1.4),
                         dpi=self.dpi)
        gridspecs = {}
        axes = {}
        
        gridspecs["gs_12345678"] = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=2,
            ncols=1,
            height_ratios=[1, 1.3],
            width_ratios=[6],
            #wspace=0.1,
            #hspace=2/3,
        )
        
        gridspecs["gs_12"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345678"][0],
            nrows=1,
            ncols=2,
            height_ratios=[2],
            width_ratios=[3, 3],
            #wspace=0.2,
            #hspace=0.25,
        )
        axes["ax_1"] = fig.add_subplot(gridspecs["gs_12"][0])
        axes["ax_2"] = fig.add_subplot(gridspecs["gs_12"][1])
        
        gridspecs["gs_3456"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_12345678"][1],
            nrows=1,
            ncols=4,
            height_ratios=[1],
            width_ratios=[0.8, 2, 2, 2],
            wspace=0.,
            #hspace=0.5,
        )

        gridspecs['gs_47'] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_3456"][1],
            nrows=2,
            ncols=1,
            height_ratios=[1, 0.06],
            width_ratios=[1],
            hspace=0.02,
        )
        gridspecs['gs_58'] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_3456"][2],
            nrows=2,
            ncols=1,
            height_ratios=[1, 0.06],
            width_ratios=[1],
            hspace=0.02,
        )
        gridspecs['gs_69'] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_3456"][3],
            nrows=2,
            ncols=1,
            height_ratios=[1, 0.06],
            width_ratios=[1],
            hspace=0.02,
        )

        gridspecs['gs_3'] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_3456"][0],
            nrows=1,
            ncols=2,
            height_ratios=[1],
            width_ratios=[0.3, 1],
            wspace=2.5,
            # hspace=0.02,
        )

        axes["ax_3"] = fig.add_subplot(gridspecs['gs_3'][0])
        axes["ax_31"] = fig.add_subplot(gridspecs['gs_3'][1])

        axes["ax_4"] = fig.add_subplot(gridspecs["gs_47"][0])
        axes["ax_5"] = fig.add_subplot(gridspecs["gs_58"][0])
        axes["ax_6"] = fig.add_subplot(gridspecs["gs_69"][0])

        axes["ax_7"] = fig.add_subplot(gridspecs["gs_47"][1])
        axes["ax_8"] = fig.add_subplot(gridspecs["gs_58"][1])
        axes["ax_9"] = fig.add_subplot(gridspecs["gs_69"][1])

        self.plot_permutation_importance(fig, axes['ax_1'])
        self.plot_position_importance(fig, axes['ax_2'])

        self.plot_avg_dist(fig, axes['ax_3'], axes['ax_31'])

        self.plot_3d_structures(fig, axes['ax_4'], axes['ax_5'], axes['ax_6'])
        self.plot_3d_distances(fig, axes['ax_7'], axes['ax_8'], axes['ax_9'])

        fig.tight_layout()
        fig.savefig('figures/manuscript_importance.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        

plotter = Plotter(plot_data)
plotter.plot()

