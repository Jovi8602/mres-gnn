import os, glob, pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from vgae import VGAEModel

def main():
    # === Configuration ===
    graphs_dir = "graphs_final/"
    max_graphs = 50
    latent_dim = 16
    hidden_dim = 64
    threshold = 0.8
    checkpoint = "saved_models/vgae_saved.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Genes
    with open('dataframes/list_of_genes.txt') as f:
        all_genes = [g.strip() for g in f.read().split(',')]

    # Load Model
    model = VGAEModel(num_node_features=10, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Sample new z vectors
    num_nodes = 100
    z_new = np.random.randn(num_nodes, latent_dim)

    # Nearest gene logic
    # Gene Latent Vector Dictionary
    gene_latents = {}
    for path in sorted(glob.glob(os.path.join(graphs_dir, "*.pkl")))[:max_graphs]:
        with open(path, 'rb') as f:
            g_x, g_edges, weights, node_ids, *_ = pickle.load(f)

        data = Data(x=torch.tensor(g_x, dtype=torch.float),
                    edge_index=torch.tensor(g_edges, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(weights, dtype=torch.float)).to(device)

        with torch.no_grad():
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar).cpu().numpy()

        for idx, vec in zip(node_ids, z):
            # Converting gene index back to name
            gene = all_genes[int(idx)]
            # Appending {gene:[latent vector]}
            gene_latents.setdefault(gene, []).append(vec)

    # Compute gene centroids and Nearest Neighbor model
    gene_names = list(gene_latents.keys())  #
    centroids = np.vstack([np.mean(vs, axis=0) for vs in gene_latents.values()]) #Mean latent vector for each gene
    nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)   #sklearn nearestneighbour searching model, a lookup object.

    idxs = nbrs.kneighbors(z_new, return_distance=False).flatten()  # Returns the index of the nearest centroid for each sampled z vector (nbrs is from gene_latent.values())
    nearest_genes = [gene_names[i] for i in idxs]                   # Looks up the gene name using the centroid index (gene_names is gene_latent.keys()). Result is a list of 100 genes, 1 for each sampled latent vector.

    # Decode for edges
    model.cpu()
    z_tensor = torch.tensor(z_new, dtype=torch.float)
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    ei_matrix = torch.vstack([row.reshape(-1), col.reshape(-1)])  # Empty dense adjacency matrix
    batch = torch.zeros(num_nodes, dtype=torch.long)        # Dummy batch, to say that tell model.decode() that all nodes belong to graph 0 (Only graph).
    with torch.no_grad():
        probs = model.decode(z_tensor, ei_matrix, batch=batch).view(num_nodes, num_nodes).cpu().numpy()

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if probs[i, j] > threshold:
                G.add_edge(i, j)

    # Draw graph and plot
    pos = nx.spring_layout(G, k=0.03, iterations=100, seed=42)
    plt.figure(figsize=(14, 8))
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='maroon')
    nx.draw_networkx_edges(G, pos, edge_color='black', width=0.5, alpha=0.7)

    # Labels (Using nearest real gene. Can be omitted).
    labels = {i: nearest_genes[i] for i in G.nodes()}   #Looking up
    y_offset = (max(y for _, y in pos.values()) - min(y for _, y in pos.values())) * 0.01
    pos_labels = {n: (x, y - y_offset) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, pos_labels, labels,
        font_size=8,
        font_color='black',
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7)
    )

    # Saving plot
    plt.title("Synthetic Genome Graph \n Annotated by Best-Matched Gene", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("vgae_synthetic_graph.png", dpi=300)

if __name__ == "__main__":
    main()