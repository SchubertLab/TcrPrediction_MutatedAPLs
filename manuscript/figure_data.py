#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

import warnings
import matplotlib as mpl
import plot_utils

warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')

#%%

class PlotData:
    # loading data is too slow to comfortably iterate on the plot
    # hence I store everything in this object and only load once
    
    def __init__(self):
        self.load_norm_data()
        self.load_unnorm_data()

    def read_data(self, is_educated, normalization='AS'):
        if is_educated:
            path_in = '../data/PH_data_educated_repertoire_and_OTI.xlsx'
        else:
            path_in = '../data/LR_data_naive_repertoire_and_OTI.xlsx'
        if normalization == 'AS':
            df = pd.read_excel(path_in, 'Normalized to initial AS')
        else:
            df = pd.read_excel(path_in, 'Unnormalized Data')
        columns = df.columns
        tcrs = columns[::4]
        columns = columns[3::4]
        df = df[columns]
        df.columns = tcrs
        return df


    def load_norm_data(self):
        data_naive = self.read_data(False)
        data_naive = data_naive.drop(['LR_OTI_1', 'LR_OTI_2'], axis=1)
        data_educated = self.read_data(True)
        data_joint = pd.concat([data_naive, data_educated], axis=1, join='inner')
        order = list(data_joint.columns)
        order.remove('OTI_PH')
        order = ['OTI_PH'] + [x for x in order if x.startswith('Ed')] + [x for x in order if not x.startswith('Ed')]
        data_joint = data_joint[order]
        self.norm_data = data_joint

    def load_unnorm_data(self):
        data_naive = self.read_data(False, normalization='None')
        data_naive = data_naive.drop(['LR_OTI_1', 'LR_OTI_2'], axis=1)
        data_educated = self.read_data(True, normalization='None')
        data_joint_unnormalized = pd.concat([data_naive, data_educated], axis=1, join='inner')
        order = list(data_joint_unnormalized.columns)
        order.remove('OTI_PH')
        order = ['OTI_PH'] + [x for x in order if x.startswith('Ed')] + [x for x in order if not x.startswith('Ed')]
        data_joint_unnormalized = data_joint_unnormalized[order]
        self.unnorm_data = data_joint_unnormalized


plot_data = PlotData()


class Plotter:
    # lazy - since I do not want to pass one hundred variables around
    # I will just set everything in self
    
    def __init__(self, plot_data):
        self.textwidth = 6.7261  # in
        self.linewidth = 3.2385  # in
        self.dpi = 350
        self.figfmt = 'tif'
        
        self.plot_data = plot_data
        
    def plot_activation_heatmap(self, fig, ax):
        data = self.plot_data.norm_data  # .transpose()
        plot = sns.heatmap(data, ax=ax,
                           cbar_kws={'label': 'Activation score', "orientation": "horizontal",
                                     'pad': 0.1})  # , cbar_pos=(0.19, 0, 0.66, .05)) #)

        plot.set_yticks([])
        plot.set_xticks([])

        plot.set(xlabel='TCRs', ylabel='APLs')
        plot.xaxis.set_label_position('bottom')
        plot.yaxis.set_label_position("left")

        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(c)')
    
    def plot_tcrs_unnormalized(self, fig, ax):
        data = []
        for col in self.plot_data.unnorm_data.columns:
            for row in self.plot_data.unnorm_data.index:
                val = self.plot_data.unnorm_data.loc[row, col]
                data.append([col, row, val])
        data = pd.DataFrame(data, columns=['TCR', 'Epitope', 'Activation'])
        data['cat'] = 'Naive'
        data.loc[data['TCR'] == 'OTI_PH', 'cat'] = 'OT1'
        data.loc[data['TCR'].str.startswith('Ed'), 'cat'] = 'Educated'

        palette = {'OT1': 'tab:green', 'Naive': 'tab:blue', 'Educated': 'tab:orange'}
        plot = sns.barplot(data=data, y='Activation', x='TCR', hue='cat', palette=palette,
                           dodge=False, errwidth=1, ci='sd',
                           ax=ax)

        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])
        plot.set(ylabel='Raw activation score')
        plot.set(xlabel=None)
        # plot.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:])
        ax.grid(False)
        sns.despine(ax=ax)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(a)')

    def plot_tcrs_normalized(self, fig, ax):
        data = []
        for col in self.plot_data.norm_data.columns:
            for row in self.plot_data.norm_data.index:
                val = self.plot_data.norm_data.loc[row, col]
                data.append([col, row, val])
        data = pd.DataFrame(data, columns=['TCR', 'Epitope', 'Activation'])
        data['cat'] = 'Naive'
        data.loc[data['TCR'] == 'OTI_PH', 'cat'] = 'OT1'
        data.loc[data['TCR'].str.startswith('Ed'), 'cat'] = 'Educated'

        palette = {'OT1': 'tab:green', 'Naive': 'tab:blue', 'Educated': 'tab:orange'}
        plot = sns.barplot(data=data, y='Activation', x='TCR', hue='cat', palette=palette, dodge=False, ci='sd',
                           errwidth=1,
                           ax=ax)
        threshold = plot.axhline(46.9, color='red', linestyle='--')
        plot.legend([threshold], ['Recruitment'])
        ax.set_ylim(bottom=0)
        plot.set_xticklabels([])

        plot.set(ylabel='Normalized activation score')
        #plot.get_legend().remove()
        ax.grid(False)
        sns.despine(ax=ax)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(b)')

    def plot_epitopes(self, fig, ax):
        data = self.plot_data.norm_data.copy()
        columns = [x for x in data.columns if x.startswith('Ed')]
        data = data[columns]
        data['Activation'] = np.mean(data.values, axis=1)
        data = data[['Activation']]
        data['Mutated position'] = [f'P{int(i / 19) + 1}' for i in range(0, 152)] + ['-']
        #data = data.reindex(['SIINFEKL'] + data.index[0:-1].values.tolist())

        activation_siinfekl = data.iloc[152]['Activation']
        data = data[:-1]

        order = ['-'] + [str(i) for i in range(1, 9)]

        palette = sns.color_palette('husl', n_colors=10)[1:]
        plot = sns.swarmplot(data=data, y='Activation', x='Mutated position', s=3,
                             hue='Mutated position', palette=palette, dodge=False, ax=ax)
        plot.set(ylabel='Activation score')
        line_siinfekl = plot.axhline(activation_siinfekl, color='red', linestyle='--')
        plot.legend([line_siinfekl], ['SIINFEKL'])

        #plot.get_legend().remove()
        ax.grid(False)
        sns.despine(ax=ax)
        plot_utils.add_axis_title_fixed_position(fig, ax, '(d)')

    def plot(self):
        sns.set(context='paper', style='whitegrid')
        plt.rc('grid', linewidth=0.3)
        sns.set_palette('colorblind')
        plot_utils.set_font_size(6)

        fig = plt.figure(figsize=(self.textwidth, self.textwidth * 0.4),
                         dpi=self.dpi)
        gridspecs = {}
        axes = {}
        
        gridspecs["gs_1234"] = mpl.gridspec.GridSpec(
            figure=fig,
            nrows=1,
            ncols=3,
            height_ratios=[1],
            width_ratios=[1.5, 1, 0.75],
            #wspace=0.1,
            #hspace=2/3,
        )
        
        gridspecs["gs_12"] = mpl.gridspec.GridSpecFromSubplotSpec(
            subplot_spec=gridspecs["gs_1234"][0],
            nrows=2,
            ncols=1,
            height_ratios=[1, 1],
            width_ratios=[5],
            #wspace=0.2,
            #hspace=0.25,
        )
        axes["ax_1"] = fig.add_subplot(gridspecs["gs_12"][1])
        axes["ax_2"] = fig.add_subplot(gridspecs["gs_12"][0])

        axes["ax_3"] = fig.add_subplot(gridspecs["gs_1234"][1])
        axes["ax_4"] = fig.add_subplot(gridspecs["gs_1234"][2])

        self.plot_activation_heatmap(fig, axes['ax_3'])
        self.plot_tcrs_unnormalized(fig, axes['ax_2'])
        self.plot_tcrs_normalized(fig, axes['ax_1'])
        self.plot_epitopes(fig, axes['ax_4'])

        fig.tight_layout()
        fig.savefig('../figures/manuscript_data.pdf',
                    dpi=self.dpi, bbox_inches='tight')
        

plotter = Plotter(plot_data)
plotter.plot()

