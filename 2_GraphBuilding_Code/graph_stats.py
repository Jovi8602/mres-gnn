#!/usr/bin/env python3

import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from pingouin import compute_effsize

def main():
    # Sorting graphs by MDR status.
    directory = 'graphs_final/'
    metadata = (pd.read_csv('dataframes/metadata.csv').dropna(subset=["name_in_presence_absence"]))

    files = glob.glob(directory + '*')
    strains = []
    for file in files:
        str = os.path.basename(file).split('.')[0]
        strains.append(str)

    mdr_status = {}

    check = 0
    yes = 0
    no = 0
    for strain in strains:
        stat = metadata.loc[metadata['ID'] == strain, 'MDR'].item()
        mdr_status[strain] = stat
        if stat == 'Yes':
            yes += 1
        else:
            no += 1

    print(f'Among downlaoded strains: {yes} with MDR, {no} without MDR.')

    graphs_sorted = {'Yes': [], 'No': []}

    for str in strains:
        if mdr_status[str] == 'Yes':
            graphs_sorted['Yes'].append(directory + f'{str}.pkl')
        else:
            graphs_sorted['No'].append(directory + f'{str}.pkl')

    print("Dictionary graphs_sorted = {'Yes': [], 'No': []} created.")

    # Creating dictionary for {'Yes/No': 'Strain': [degree],[length],[cluster],[weight]}
    stats_sorted = {'Yes': {}, 'No': {}}

    for i in range(2):
        if i == 0:
            mdr = 'Yes'
        else:
            mdr = 'No'
        for graph in graphs_sorted[mdr]:
            strain = graph.split('/')[-1].split('.')[0]
            length = []
            degree = []
            centrality = []
            cluster = []
            with open(graph, 'rb') as f:
                nodes, edges, weights, genes, node_metrics, graph_metrics = pickle.load(f)
            for node in nodes:
                length.append(node[8])
            for node in node_metrics:
                degree.append(node[0])
                centrality.append(node[1])
                cluster.append(node[2])
            stats_sorted[mdr][strain] = {}
            stats_sorted[mdr][strain]['degree'] = list(degree)
            stats_sorted[mdr][strain]['length'] = list(length)
            stats_sorted[mdr][strain]['centrality'] = list(centrality)
            stats_sorted[mdr][strain]['cluster'] = list(cluster)
            stats_sorted[mdr][strain]['weight'] = weights
            stats_sorted[mdr][strain]['size'] = graph_metrics[1]
            stats_sorted[mdr][strain]['components'] = graph_metrics[3]
            stats_sorted[mdr][strain]['assortivity'] = graph_metrics[2]

    print(
        'Dictionary stats_sorted = {Yes/No: Strain: [Degree], [Length], [Centrality], [Cluster], [Weights], [Size], [Components], [Assortivity]} created.')

    statistics = ['node_num', 'edge_num', 'length', 'degree', 'centrality', 'cluster', 'weight', 'size', 'components',
                  'assortivity']

    with PdfPages('MDR_graph_stats.pdf') as pdf:
        for stat in statistics:
            yes_list, no_list = [], []
            for strain in stats_sorted['Yes']:
                if stat == 'node_num':
                    avg = len(stats_sorted['Yes'][strain]['degree'])
                elif stat == 'edge_num':
                    avg = len(stats_sorted['Yes'][strain]['weight'])
                elif stat == 'components' or stat == 'assortivity' or stat == 'size':
                    avg = stats_sorted['Yes'][strain][stat]
                else:
                    avg = np.mean(stats_sorted['Yes'][strain][stat])
                yes_list.append(avg)

            for strain in stats_sorted['No']:
                if stat == 'node_num':
                    avg = len(stats_sorted['No'][strain]['degree'])
                elif stat == 'edge_num':
                    avg = len(stats_sorted['No'][strain]['weight'])
                elif stat == 'components' or stat == 'assortivity' or stat == 'size':
                    avg = stats_sorted['No'][strain][stat]
                else:
                    avg = np.mean(stats_sorted['No'][strain][stat])
                no_list.append(avg)

            # page 1: KDE plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.kdeplot(yes_list, label=f'MDR+ (n={len(yes_list)})',
                        fill=True, alpha=0.35, common_norm=False, ax=ax)
            sns.kdeplot(no_list, label=f'MDR– (n={len(no_list)})',
                        fill=True, alpha=0.35, common_norm=False, ax=ax)
            ax.set_xlabel(f'Mean {stat} per graph')
            ax.set_ylabel('Probability density')
            ax.set_title(f'{stat} distribution')
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # page 2: numerical summary
            U, p = stats.mannwhitneyu(yes_list, no_list, alternative='two-sided')
            delta = compute_effsize(yes_list, no_list)

            summary = (
                f'{stat.upper()}   (per-graph averages)\n'
                '──────────────────────────────────────\n'
                f'MDR+  (n={len(yes_list):3d})  mean = {np.mean(yes_list):,.2f} '
                f' median = {np.median(yes_list):,.2f}\n'
                f'MDR–  (n={len(no_list):3d})  mean = {np.mean(no_list):,.2f} '
                f' median = {np.median(no_list):,.2f}\n\n'
                f'Mann–Whitney U = {U:.0f}\n'
                f' two-sided p   = {p:.2e}\n'
                f'Cliff’s Δ      = {delta:+.3f}\n'
            )

            fig2 = plt.figure(figsize=(8.5, 4))
            fig2.text(0.02, 0.98, summary, va='top', family='monospace', size=11)
            pdf.savefig(fig2)
            plt.close(fig2)

    print('PDF saved as MDR_graph_stats.pdf')

if __name__ == "__main__":
    main()
