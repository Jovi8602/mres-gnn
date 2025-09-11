#!/usr/bin/env python3
import argparse
import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torch_geometric.nn import global_mean_pool

from graph_classifier import (prepare_data,prepare_splits,GNNClassifier_GIN,GNNClassifier_GIN_embed,GNNClassifier_GCN,GNNClassifier_GCN_embed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gnn_model", type=int, default=1, help="Which GNN architecture to use")
    args = parser.parse_args()

    if args.gnn_model == 1:
        arch = "GNNClassifier_GCN"
    if args.gnn_model == 2:
        arch = "GNNClassifier_GIN"
    if args.gnn_model == 3:
        arch = "GNNClassifier_GCN_embed"
    if args.gnn_model == 4:
        arch = "GNNClassifier_GCN_embed"
    # paths
    GRAPH_DIR = "graphs_final/"
    TARGET_JSON = "dataframes/mdr1000.json"
    MODEL_PATH = f"saved_models/{arch}.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model set up. Hyperparameters must match saved model.
    data_list = prepare_data(GRAPH_DIR, TARGET_JSON)
    _, loader_test = prepare_splits(data_list, seed=51, train_ratio=0.8, batch_size=32)
    num_node_features = data_list[0].x.size(1)

    if args.gnn_model == 1:
        model = GNNClassifier_GCN(num_node_features, hidden_dim=256, output_dim=2).to(DEVICE)
    if args.gnn_model == 2:
        model = GNNClassifier_GIN(num_node_features, hidden_dim=256, output_dim=2).to(DEVICE)
    if args.gnn_model == 3:
        model = GNNClassifier_GCN_embed(num_node_features+1024, hidden_dim=256, output_dim=2).to(DEVICE)
    if args.gnn_model == 4:
        model = GNNClassifier_GIN_embed(num_node_features+1024, hidden_dim=256, output_dim=2).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # === Extract embeddings (pre-softmax) ===
    all_embeddings = []
    all_labels = []

    if args.gnn_model == 1:
        with torch.no_grad():
            for data in loader_test:
                data = data.to(DEVICE)
                edge_weight = data.edge_attr

                x = data.x
                x = model.conv1(x, data.edge_index, edge_weight=edge_weight)
                x = torch.relu(x)
                x = model.dropout(x)

                x = model.conv2(x, data.edge_index, edge_weight=edge_weight)
                x = torch.relu(x)
                x = model.dropout(x)

                pooled = global_mean_pool(x, data.batch)
                all_embeddings.append(pooled.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    elif args.gnn_model == 3:
        with torch.no_grad():
            for data in loader_test:
                data = data.to(DEVICE)
                node_ids = data.node_ids
                edge_weight = data.edge_attr

                embed = model.embedding(node_ids)
                x = torch.cat([data.x, embed], dim=1)

                x = model.conv1(x, data.edge_index, edge_weight=edge_weight)
                x = torch.relu(x)
                x = model.dropout(x)

                x = model.conv2(x, data.edge_index, edge_weight=edge_weight)
                x = torch.relu(x)
                x = model.dropout(x)

                pooled = global_mean_pool(x, data.batch)
                all_embeddings.append(pooled.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    elif args.gnn_model == 2:
        with torch.no_grad():
            for data in loader_test:
                data = data.to(DEVICE)

                x = model.conv1(data.x, data.edge_index)
                x = model.bn1(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv2(x, data.edge_index)
                x = model.bn2(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv3(x, data.edge_index)
                x = model.bn3(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv4(x, data.edge_index)
                x = model.bn4(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                pooled = global_mean_pool(x, data.batch)
                all_embeddings.append(pooled.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    elif args.gnn_model == 4:
        with torch.no_grad():
            for data in loader_test:
                data = data.to(DEVICE)
                node_ids = data.node_ids
                embed = model.embedding(node_ids)
                x = torch.cat([data.x, embed], dim=1)

                x = model.conv1(data.x, data.edge_index)
                x = model.bn1(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv2(x, data.edge_index)
                x = model.bn2(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv3(x, data.edge_index)
                x = model.bn3(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                x = model.conv4(x, data.edge_index)
                x = model.bn4(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
                x = model.dropout(x)

                pooled = global_mean_pool(x, data.batch)
                all_embeddings.append(pooled.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    # === Stack & Reduce ===
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)

    # === Get soft colors from coolwarm colormap ===
    cmap = cm.get_cmap("coolwarm")
    color_mdr_minus = cmap(0.1)
    color_mdr_plus = cmap(0.9)

    # === Plot with soft color legend ===
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], c=[color_mdr_minus], label="MDR–", alpha=0.7)
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], c=[color_mdr_plus], label="MDR+", alpha=0.7)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"UMAP of Graph Representations {arch[14:]} ")
    plt.legend(title="MDR Status")
    plt.tight_layout()
    plt.savefig("umap_projection_gin_softcolors.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
