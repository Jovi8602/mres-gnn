#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import os
import glob
import numpy as np
import pickle
import community as community_louvain
from Bio import SeqIO

def generate_graph(dataframe):
    df = pd.read_csv(dataframe)
    strain = os.path.basename(dataframe).split("_results")[0]
    with open('dataframes/list_of_genes.txt', 'r') as f:
        gene_list = f.read()
    genes = [item.strip() for item in gene_list.split(',')]

    fasta = f'assembled/{strain}.fasta'
    size = sum(len(record.seq) for record in SeqIO.parse(fasta, "fasta"))

    df['subject_start'] = pd.to_numeric(df['subject_start'], errors='coerce')
    df['subject_end'] = pd.to_numeric(df['subject_end'], errors='coerce')
    df['midpoint'] = (df['subject_start'] + df['subject_end']) / 2
    df['start'] = df[['subject_start', 'subject_end']].min(axis=1)
    df['end'] = df[['subject_start', 'subject_end']].max(axis=1)
    df['length'] = pd.to_numeric(df['length'] / 1000, errors='coerce')
    df['direction'] = df.apply(lambda row: 1 if row['subject_start'] < row['subject_end'] else -1, axis=1)

    G = nx.Graph()

    # Node attributes
    for _, row in df.iterrows():
        gene_id = row['query']
        coords = rotary_embedding(row['midpoint'])
        G.add_node(
            gene_id,
            name=genes.index(gene_id),
            length=row['length'],
            position=coords,
            strand=row['direction']
        )

    # Edge building
    nodes_data = [(row['query'], row['start'], row['end'], row['midpoint'], row['direction'])
                  for _, row in df.iterrows()]
    nodes_data.sort(key=lambda x: x[3])

    threshold = 10000
    tau = 2000

    for i in range(len(nodes_data)):
        for j in range(i + 1, len(nodes_data)):
            node_i, start_i, end_i, mid_i, dir_i = nodes_data[i]
            node_j, start_j, end_j, mid_j, dir_j = nodes_data[j]

            if max(start_i, start_j) <= min(end_i, end_j):
                G.add_edge(node_i, node_j, weight=1.0)
                continue

            gap = abs(mid_i - mid_j)
            dist = min(gap, size - gap)
            if dist <= threshold:
                wt = 1 / (1 + (dist / tau))
                G.add_edge(node_i, node_j, weight=wt)

    # Node metrics
    degrees = nx.degree(G)
    centrality_values = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    clustering_wt = nx.clustering(G, weight='weight')

    # Graph-level metrics
    assort = nx.degree_assortativity_coefficient(G)
    components = list(nx.connected_components(G))
    num_components = len(components)
    partition = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(partition, G, weight='weight')

    for node in G.nodes():
        G.nodes[node]['degree'] = degrees[node]
        G.nodes[node]['centrality'] = centrality_values[node]
        G.nodes[node]['cluster'] = clustering[node]
        G.nodes[node]['cluster_wt'] = clustering_wt[node]

    G.graph['strain'] = strain
    G.graph['size'] = size
    G.graph['assort'] = assort
    G.graph['components'] = num_components
    G.graph['modularity'] = modularity

    return G

def rotary_embedding(coord, dim=8, base=10000):
    half_dim = dim // 2
    freq = np.exp(-np.log(base) * np.arange(half_dim) / half_dim)
    angles = coord * freq
    emb = np.concatenate([np.sin(angles), np.cos(angles)])
    return emb

def save_graph(G, filename):
    nodes = list(G.nodes())
    g_x = [list(G.nodes[node]['position']) + [G.nodes[node]['length'], G.nodes[node]['strand']] for node in nodes]
    x_metrics = [[G.nodes[node]['degree'], G.nodes[node]['centrality'], G.nodes[node]['cluster'], G.nodes[node]['cluster_wt']] for node in nodes]
    node_names = [G.nodes[node]['name'] for node in nodes]

    node_to_index = {node: i for i, node in enumerate(nodes)}
    edges = []
    weights = []
    for u, v, attr in G.edges(data=True):
        edges.append([node_to_index[u], node_to_index[v]])
        edges.append([node_to_index[v], node_to_index[u]])
        weights.append(attr.get('weight', 1))
        weights.append(attr.get('weight', 1))

    g_x = np.array(g_x)
    x_metrics = np.array(x_metrics)
    g_edges = np.array(edges)
    weights = np.array(weights)

    strain = G.graph.get('strain')
    size = G.graph.get('size')
    assort = G.graph.get('assort')
    components = G.graph.get('components')
    modularity = G.graph.get('modularity')
    g_metrics = [strain, size, assort, components, modularity]

    with open(filename, 'wb') as f:
        pickle.dump((g_x, g_edges, weights, node_names, x_metrics, g_metrics), f)

    print(f"Graph saved to {filename}")

def main():
    input_pattern = "results/*"
    out_dir = "graphs_final"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"[run] No result files found at: {input_pattern}")
        return

    for df_path in files:
        try:
            strain = os.path.basename(df_path).split("_results")[0]
            out_path = os.path.join(out_dir, f"{strain}.pkl")

            print(f"\n[run] Building graph for: {strain}")
            print(f"[run] Input : {df_path}")
            print(f"[run] Output: {out_path}")

            G = generate_graph(df_path)
            save_graph(G, out_path)
        except Exception as e:
            print(f"[warn] Failed on {df_path}: {e}")

if __name__ == "__main__":
    main()