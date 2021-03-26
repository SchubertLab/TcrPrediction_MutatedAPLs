import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def create_activation_histogram(data, do_logit=False):
    set_style()
    activation_values = data.to_numpy().flatten()
    if do_logit:
        activation_values = logit(activation_values)

    sns.distplot(activation_values, kde=False)
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Activation')
    plt.savefig('../results/visualisation/histogram.png')


def create_box_plots(data, do_logit=False):
    set_style()
    # activation_values = data.to_numpy()
    if do_logit:
        data = logit(data)
    sns.boxplot(data=data)
    # plt.boxplot(activation_values)
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('TCR')
    plt.xticks(rotation=45)
    plt.ylabel('Activation')
    plt.savefig('../results/visualisation/box_plots.png')


# def create_cumulative_distribution(data, do_logit=False):
#     set_style()
#     activation_values = data.to_numpy().flatten()
#     if do_logit:
#         activation_values = logit(activation_values)
#     activation_values = np.sort(activation_values)
#     n = activation_values.shape[0]
#     y = np.arange(1, n+1, dtype=np.float) / n
#
#     sns.ecdfplot(activation_values)
#     plt.plot(activation_values, y)
#     plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Activation')
    # plt.ylabel('n')
    # plt.show()


def set_style():
    plt.clf()
    # sns.set_theme()
    # axis = plt.axes()
    # axis.spines['top'].set_visible(False)
    # axis.spines['right'].set_visible(False)
    # axis.spines['left'].set_visible(False)
    pass

def logit(x):
    x /= 100
    x = np.log(x/(1-x) + 0.0001)
    return x


if __name__ == "__main__":
    data = pd.read_csv('../data/tcrs_high.csv', index_col=0)
    create_activation_histogram(data)
    create_box_plots(data)
    # create_cumulative_distribution(data)
