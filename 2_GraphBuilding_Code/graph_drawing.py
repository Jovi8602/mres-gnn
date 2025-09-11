import os
import pandas as pd
import networkx as nx
from Bio import SeqIO
import matplotlib.pyplot as plt

def draw_graph(dataframe):
    threshold = 10000  # bp
    tau = 2000  # decay constant for weight

    df = pd.read_csv(dataframe)
    strain = os.path.basename(dataframe).split("_results")[0]

    # Uses the same graph construction logic
    fasta = f'assembled/{strain}.fasta'
    size = sum(len(record.seq) for record in SeqIO.parse(fasta, "fasta"))

    df['subject_start'] = pd.to_numeric(df['subject_start'], errors='coerce')
    df['subject_end']   = pd.to_numeric(df['subject_end'], errors='coerce')
    df['start']   = df[['subject_start', 'subject_end']].min(axis=1)
    df['end']     = df[['subject_start', 'subject_end']].max(axis=1)
    df['midpoint'] = (df['subject_start'] + df['subject_end']) / 2

    G = nx.Graph()
    for gene_id in df['query']:
        G.add_node(gene_id)

    nodes_data = list(zip(df['query'], df['start'], df['end'], df['midpoint']))
    nodes_data = [t for t in nodes_data if pd.notna(t[1]) and pd.notna(t[2]) and pd.notna(t[3])]
    nodes_data.sort(key=lambda x: x[3])

    for i in range(len(nodes_data)):
        node_i, start_i, end_i, mid_i = nodes_data[i]
        for j in range(i + 1, len(nodes_data)):
            node_j, start_j, end_j, mid_j = nodes_data[j]

            if max(start_i, start_j) <= min(end_i, end_j):
                G.add_edge(node_i, node_j, weight=1.0)
                continue

            gap = abs(mid_i - mid_j)
            dist = min(gap, size - gap)
            if dist <= threshold:
                wt = 1 / (1 + (dist / tau))
                G.add_edge(node_i, node_j, weight=wt)

    # Draw and Save
    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(G, k=0.03, iterations=100, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='slateblue')
    nx.draw_networkx_edges(G, pos, edge_color='black', width=0.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.tight_layout()
    plt.savefig(f"genomic_graph.png", dpi=300)
    plt.close()

# Run it (Any strain works)
data = 'results/ESC_BA6745AA_AS_results.csv'
draw_graph(data)
