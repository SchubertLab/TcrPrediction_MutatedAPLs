import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP


def get_data():
    data = pd.read_csv('../data/tcrs_high.csv', index_col=0)
    data = data.transpose()
    # data = data.transpose()
    return data


def tsne_transform(data, perplexity=5):
    data_embedded = TSNE(perplexity=perplexity).fit_transform(data)
    df = pd.DataFrame()
    df['x'] = data_embedded[:, 0]
    df['y'] = data_embedded[:, 1]

    df['label'] = data.index
    if len(df['label'][0]) == 8:
        df['label'] = shorten_label(df['label'])
    return df


def shorten_label(labels):
    new_labels = []
    for label in labels:
        for let_siinfekl, let_label in zip('SIINFEKL', label):
            if let_siinfekl != let_label:
                new_labels.append(let_label)
    new_labels.append('X')
    return new_labels


def umap_transform(data, neighbors=5, min_dist=0.):
    data_embedded = UMAP(n_neighbors=neighbors, min_dist=min_dist).fit_transform(data)
    df = pd.DataFrame()
    df['x'] = data_embedded[:, 0]
    df['y'] = data_embedded[:, 1]
    df['label'] = data.index
    if len(df['label'][0]) == 8:
        df['label'] = shorten_label(df['label'])
    return df


def pca_transform(data):
    data_embedded = PCA(n_components=2).fit_transform(data)
    df = pd.DataFrame()
    df['x'] = data_embedded[:, 0]
    df['y'] = data_embedded[:, 1]
    df['label'] = data.index
    if len(df['label'][0]) == 8:
        df['label'] = shorten_label(df['label'])
    return df


def plot_2d_data(data_reduced, path_save=None, with_text=True, colors=None):
    plt.clf()
    plt.subplots(figsize=(11, 7))
    n_colors = 1
    palette = 'tab10'
    if colors is not None:
        n_colors = len(set(colors))
        if n_colors > 10:
            palette = 'Spectral'
    plot = sns.scatterplot(data=data_reduced, x='x', y='y', hue=colors, palette=sns.color_palette(palette, n_colors))
    plot.set_facecolor('w')
    if colors is not None:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if with_text:
        for i, row in data_reduced.iterrows():
            plot.text(row['x']+0.02, row['y'], row['label'])
    if path_save is None:
        plt.show()
    else:
        plt.savefig(path_save, bbox_inches='tight')
    plt.close()


def hyperparameter_tsn(use_label=True):
    data = get_data()
    path_out = '../results/visualisation/tsne/'
    for i in range(1, 50):
        data_red = tsne_transform(data, perplexity=i)
        # groups = create_positional_grouping
        groups = create_positional_grouping(153)
        plot_2d_data(data_red, path_save=path_out+str(i)+'.png', colors=groups, with_text=use_label)


def hyperparameter_umap(use_label=True):
    data = get_data()
    path_out = '../results/visualisation/umap/'
    for i in range(2, 50):
        for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99):
            data_red = umap_transform(data, i, d)
            groups = create_positional_grouping(153)
            plot_2d_data(data_red, path_save=path_out + str(i) + '_' + str(d) + '.png', colors=groups,
                         with_text=use_label)


def pca_plot(use_label=True):
    data = get_data()
    path_out = '../results/visualisation/pca/0.png'
    data_red = pca_transform(data)
    groups = create_positional_grouping(153)
    plot_2d_data(data_red, path_out, colors=groups, with_text=use_label)


def create_positional_grouping(length):
    subgroups = [int(idx/19)+1 for idx in list(range(length))]
    subgroups = ['Position ' + str(el) for el in subgroups]
    subgroups[-1] = 'SIINFEKL'
    return subgroups


if __name__ == '__main__':
    # hyperparameter_tsn(use_label=False)
    # pca_plot(use_label=False)
    # hyperparameter_umap(use_label=False)
    data_org = get_data()
    epitope_data_red = tsne_transform(data_org, perplexity=5)
    # epitope_data_red = umap_transform(data_org, neighbors=3, min_dist=0.8)
    # epitope_data_red = pca_transform(data_org)
    group_position = create_positional_grouping(153)
    group_aa = create_positional_grouping(153)
    plot_2d_data(epitope_data_red, with_text=True, colors=None, path_save='tcr.png')
    # plot_2d_data(epitope_data_red, with_text=True, colors=group_position, path_save='pos_annotated.png')
    # plot_2d_data(epitope_data_red, with_text=False, colors=group_aa, path_save='aa.png')
