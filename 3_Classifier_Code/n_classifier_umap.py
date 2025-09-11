import os
import glob
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import argparse

from node_classifier import prepare_data, prepare_splits, Node_GCN, Node_GIN, Node_GCN_Embed, Node_GIN_Embed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gnn_model", type=int, default=1, help="Which GNN architecture to use")
    args = parser.parse_args()

    #Recreating splits
    graph_directory = "graphs_final/"
    seed = 51
    train_ratio = 0.8
    data_list = prepare_data(graph_directory)
    _, loader_test = prepare_splits(data_list, seed=seed, train_ratio=train_ratio, batch_size=1)

    #Getting just 3 graphs to view node representations
    graph_list = []
    for i, data in enumerate(loader_test):
        if i >= 3:
            break
        graph_list.append(data)

    #Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features = data_list[0].num_node_features

    if args.gnn_model == 1:
        model = Node_GCN(num_node_features, hidden_dim=256, output_dim=2).to(device)
    if args.gnn_model == 2:
        model = Node_GIN(num_node_features, hidden_dim=256, output_dim=2).to(device)
    if args.gnn_model == 3:
        model = Node_GCN_Embed(num_node_features, hidden_dim=256, output_dim=2).to(device)
    if args.gnn_model == 4:
        model = Node_GIN_Embed(num_node_features, hidden_dim=256, output_dim=2).to(device)

    if args.gnn_model == 1:
        arch = "Node_GCN"
    if args.gnn_model == 2:
        arch = "Node_GIN"
    if args.gnn_model == 3:
        arch = "Node_GCN_Embed"
    if args.gnn_model == 4:
        arch = "Node_GIN_Embed"

    checkpoint = torch.load(f"saved_models/{arch}.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_embeddings = []
    all_labels = []

    if args.gnn_model == 1:
        with torch.no_grad():
            for data in graph_list:
                data = data.to(device)
                h = model.conv1(data.x, data.edge_index, edge_weight=data.edge_attr)
                h = torch.relu(h)
                h = model.dropout(h)
                h = model.conv2(h, data.edge_index, edge_weight=data.edge_attr)
                h = torch.relu(h)
                h = model.dropout(h)

                all_embeddings.append(h.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())

    if args.gnn_model == 3:
        with torch.no_grad():
            for data in graph_list:
                data = data.to(device)
                emb = model.embedding(data.node_ids)
                h = torch.cat([data.x, emb], dim=1)
                h = model.conv1(h, data.edge_index, edge_weight=data.edge_attr)
                h = torch.relu(h)
                h = model.dropout(h)
                h = model.conv2(h, data.edge_index, edge_weight=data.edge_attr)
                h = torch.relu(h)
                h = model.dropout(h)

                all_embeddings.append(h.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())

    if args.gnn_model == 2:
        with torch.no_grad():
            for data in graph_list:
                data = data.to(device)
                h = model.conv1(data.x, data.edge_index)
                h = torch.relu(h)
                h = model.dropout(h)
                h = model.conv2(h, data.edge_index)
                h = torch.relu(h)
                h = model.dropout(h)
                all_embeddings.append(h.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())

    if args.gnn_model == 4:
        with torch.no_grad():
            for data in graph_list:
                data = data.to(device)
                emb = model.embedding(data.node_ids)
                h = torch.cat([data.x, emb], dim=1)
                h = model.conv1(h, data.edge_index)
                h = torch.relu(h)
                h = model.dropout(h)
                h = model.conv2(h, data.edge_index)
                h = torch.relu(h)
                h = model.dropout(h)
                all_embeddings.append(h.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    # === 6) UMAP projection ===
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(embeddings)

    # === 7) Plot ===
    plt.figure(figsize=(8, 6))

    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1],
                c='gray', s=5, alpha=0.2, label='Non-essential')

    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1],
                c='indianred', s=10, alpha=1.0, label='Essential')

    plt.title(f"UMAP of {arch[5:]} Node Embeddings\nColored by Essentiality", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    print('Done?')
    plt.savefig("Classifier_Essentiality_UMAP.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

