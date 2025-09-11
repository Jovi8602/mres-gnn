#!/usr/bin/env python3

import os, glob, pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
import pandas as pd
from torch_geometric.data import Data
from vgae import VGAEModel

# === Configuration (mirrors your newer script) ===
graphs_dir  = "graphs_final/"
num_graphs  = 5
hidden_dim  = 64
latent_dim  = 16
checkpoint  = "saved_models/vgae_saved.pth"
random_seed = 44

def main():


    # Load Data
    with open('dataframes/list_of_genes.txt') as f:
        all_genes = [g.strip() for g in f.read().split(',')]
    keio_df = pd.read_excel('dataframes/keio_collection.xlsx', header=None)
    essential_names = set(keio_df[0].astype(str).tolist())
    essential_idx = {all_genes.index(g) for g in essential_names if g in all_genes}

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAEModel(num_node_features=10, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Gather latent vectors
    emb_list, label_list, length_list, gene_id_list = [], [], [], []

    paths = sorted(glob.glob(os.path.join(graphs_dir, "*.pkl")))[:num_graphs]
    with torch.no_grad():
        for path in paths:
            with open(path, 'rb') as f:
                g_x, g_edges, weights, node_ids, *rest = pickle.load(f)

            data = Data(
                x=torch.tensor(g_x, dtype=torch.float),
                edge_index=torch.tensor(g_edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(weights, dtype=torch.float)
            ).to(device)

            # latent z vectors
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar).cpu().numpy()
            emb_list.append(z)

            # flatten node IDs
            flat_ids = [nid[0] if isinstance(nid, (list, tuple)) and len(nid) == 1 else nid for nid in node_ids]
            flat_ids = np.asarray(flat_ids, dtype=int)
            gene_id_list.append(flat_ids)

            # essentiality labels
            y = np.isin(flat_ids, list(essential_idx)).astype(np.int64)
            label_list.append(y)

            # gene lengths
            lengths = np.asarray(g_x, dtype=np.float32)[:, 8]
            length_list.append(lengths)

    embeddings = np.vstack(emb_list)
    labels     = np.concatenate(label_list)
    lengths    = np.concatenate(length_list)
    gene_ids = np.concatenate(gene_id_list)

    # UMAP Projection
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_seed)
    proj = reducer.fit_transform(embeddings)

    # Essentiality UMAP Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1],
                c='gray', s=5, alpha=0.25, label='Non-essential')
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1],
                c='indianred', s=10, alpha=0.95, label='Essential')
    plt.title("UMAP of VGAE Latent Space\nColored by Essentiality", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("vgae_umap_essentiality.png", dpi=300)
    plt.close()

    # Length UMAP Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1],
                     c=lengths, cmap='copper', s=5, alpha=0.7)
    plt.title("UMAP of VGAE Latent Space\nColored by Gene Length", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    cbar = plt.colorbar(sc)
    cbar.set_label("Gene Length (kb)")
    plt.tight_layout()
    plt.savefig("vgae_umap_length.png", dpi=300)
    plt.close()

    # ID UMAP Plot
    # ---------- Plot 3: Selected gene identities ----------
    # Option A: use MANUAL_GENE_NAMES (filter to those that exist)
    MANUAL_GENE_NAMES = ['epmC', 'lpxK', 'rplB', 'flgK', 'mglA_2']
    manual_selected = [all_genes.index(g) for g in MANUAL_GENE_NAMES if g in all_genes]

    # Option B: if manual list is empty, fallback to most frequent genes
    selected = manual_selected
    if len(selected) == 0:
        counts = np.bincount(gene_ids)
        eligible = np.where(counts >= 5)[0]  # “common” genes threshold
        rng = np.random.default_rng(random_seed)
        pick_n = min(5, len(eligible))
        selected = rng.choice(eligible, size=pick_n, replace=False).tolist()

    plt.figure(figsize=(8, 6))
    mask_non = ~np.isin(gene_ids, selected)
    plt.scatter(proj[mask_non, 0], proj[mask_non, 1], c='lightgray', s=5, alpha=0.3, zorder=1)

    cmap = plt.get_cmap('tab10')
    patches = []
    for idx, gid in enumerate(selected):
        mask_sel = gene_ids == gid
        color = cmap(idx % 10)
        plt.scatter(proj[mask_sel, 0], proj[mask_sel, 1], c=[color], s=20, alpha=0.85, zorder=2)
        patches.append(mpatches.Patch(color=color, label=all_genes[gid] if gid < len(all_genes) else f"id:{gid}"))

    plt.legend(handles=patches, title="Genes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("UMAP of VGAE Latent Space\nColored by Selected Gene Identity", fontsize=14)
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("vgae_umap_id.png", dpi=300)
    plt.close()

    print("Saved: "
          "vgae_latents_umap_essentiality.png, "
          "vgae_latents_umap_length.png, "
          "vgae_latent_by_gene_umap.png")

if __name__ == "__main__":
    main()
