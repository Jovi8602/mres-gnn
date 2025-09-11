import argparse
import gc
import glob
import json
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (BatchNorm, GCNConv, GINConv, global_mean_pool)

from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score)


def get_graph(graph, target):
    with open(graph, 'rb') as f:
        g_x, g_edges, weights, node_ids, x_metric, g_metric = pickle.load(f)

    x = torch.tensor(g_x, dtype=torch.float)             #Shape = [num_nodes, num_features]
    edge_index = torch.tensor(g_edges, dtype=torch.long) #Shape = [2, num_edges]
    edge_attr = torch.tensor(weights, dtype=torch.float) #Shape = [num_edges]
    y = torch.tensor([1 if gene in target else 0 for gene in node_ids], dtype=torch.long)
    node_ids = torch.tensor(node_ids, dtype=torch.long)  #Shape = [num_nodes]
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, node_ids=node_ids, y=y)
    return data

def prepare_data(graphs_dir):
    pattern = os.path.join(graphs_dir, "*.pkl")
    file_list = sorted(glob.glob(pattern))

    keio = (pd.read_excel('dataframes/keio_collection.xlsx', header=None))[0].astype(str).tolist()
    with open('dataframes/list_of_genes.txt') as f:
        all_genes = [g.strip() for g in f.read().split(',')]

    essentials = []
    for gene in keio:
        if gene in all_genes:
            index = all_genes.index(gene)
            essentials.append(index)

    print(essentials)

    data_list = []
    for file in file_list:
        data = get_graph(file, essentials)
        data_list.append(data)

    print(f'Total number of graphs: {len(data_list)}')

    return data_list

def prepare_splits(data_list, seed, train_ratio, batch_size):
    np.random.seed(seed)
    np.random.shuffle(data_list)
    split_idx = int(train_ratio * len(data_list))
    train_list = data_list[:split_idx]
    test_list = data_list[split_idx:]

    print(f'Number of training graphs: {len(train_list)}')
    print(f'Number of test graphs: {len(test_list)}')

    loader_train = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_list, batch_size=batch_size, shuffle=True)
    print("Number of batches:", len(loader_train))
    print("Total graphs in loader_train:", len(loader_train.dataset))

    return loader_train, loader_test

def calculate_metrics(preds, probs, labels):
    all_preds = torch.cat(preds).numpy()
    all_probs = torch.cat(probs).numpy()
    all_labels = torch.cat(labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    prauc = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return auc, prauc, f1, cm

def train(model, loader_train, loader_test, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    total_nodes = 0
    correct = 0

    for data in loader_train:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)                   # [N, 2]
        loss = criterion(out, data.y)       # CrossEntropyLoss expects [N,2], [N]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_nodes  #loss.item() gives average loss this batch, we don't want that.
        total_nodes += data.num_nodes

        preds = out.argmax(dim=1)
        correct += (preds == data.y).sum().item()

    loss_train = total_loss / total_nodes
    acc_train = correct / total_nodes

    model.eval()
    total_loss = 0
    total_nodes = 0
    correct = 0

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)
            out = model(data)                   # [N,2]
            loss = criterion(out, data.y)

            total_loss += loss.item() * data.num_nodes
            total_nodes += data.num_nodes

            probs = F.softmax(out, dim=1)[:,1]  # P(class=1)
            preds = out.argmax(dim=1)           # [N]

            correct += (preds == data.y).sum().item()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(data.y.cpu())

        loss_test = total_loss / total_nodes
        acc_test = correct / total_nodes

        scheduler.step(loss_test)

        # reuse your calculate_metrics (expects lists or tensors)
        auc, prauc, f1, cm = calculate_metrics(all_preds, all_probs, all_labels)

    return loss_train, acc_train, loss_test, acc_test, auc, prauc, f1, cm

def train_loop(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs):
    """Train the model for a number of epochs and return the loss and accuracy history."""
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        loss_train, acc_train, loss_test, acc_test, auc, prauc, f1, cm = train(model, loader_train, loader_test, optimizer, criterion, scheduler, device)
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        print(f'Epoch: {epoch}, Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.4f}, ' +
              f'Test Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm

def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm):
    #Plot loss, accuracy, and final AUC/F1 score into a single multi-page PDF
    num_epochs = len(train_losses)

    with PdfPages('training_results.pdf') as pdf:
        # Page 1: Loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss over Epochs')
        plt.legend()
        pdf.savefig()
        plt.close()

        # Page 2: Accuracy plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy over Epochs')
        plt.legend()
        pdf.savefig()
        plt.close()

        # Page 3: Metrics
        metrics_dict = {'AUC': auc, 'F1 Score': f1}
        plt.figure(figsize=(8, 5))
        metric_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
        cm_text = f"\nConfusion Matrix:\n{cm}"
        metric_text += cm_text
        plt.text(0.01, 0.99, metric_text, fontsize=16, ha='left', va='top', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Final Evaluation Metrics', loc='left')
        pdf.savefig()
        plt.close()

class Node_GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super(Node_GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final linear for node-level classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr  # should be shape [num_edges]

        # First GCN layer with weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer with weights
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Node-level predictions
        out = self.fc(x)
        return out

class Node_GCN_Embed(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super(Node_GCN_Embed, self).__init__()
        self.embedding = nn.Embedding(55049, embedding_dim)

        self.conv1 = GCNConv(num_node_features + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final linear for node-level classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr  # should be shape [num_edges]
        node_ids = data.node_ids  # if using node_id embeddings

        # Concatenate feature + ID embedding
        embed = self.embedding(node_ids)
        x = torch.cat([x, embed], dim=1)

        # First GCN layer with weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer with weights
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Node-level predictions
        out = self.fc(x)
        return out

class Node_GIN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super().__init__()
        # two‐layer GCN
        mlp1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(mlp1)

        # MLP for second GINConv
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(mlp2)

        # final node‑level classifier
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # first GIN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # second GIN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # per‑node logits
        out = self.lin(x)
        return out

class Node_GIN_Embed(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super().__init__()
        # ID Embeddings
        self.embedding = nn.Embedding(55049, embedding_dim)

        # two‐layer GCN
        mlp1 = nn.Sequential(
            nn.Linear(num_node_features+embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(mlp1)

        # MLP for second GINConv
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(mlp2)

        # final node‑level classifier
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        emb = self.embedding(data.node_ids)
        x   = torch.cat([x, emb], dim=1)

        # first GIN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # second GIN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # per‑node logits
        out = self.lin(x)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_dir", type=str, default="graphs_20k/", help="Path to graph pickle files")
    parser.add_argument("gnn_model", type=int, default=1, help="Which GNN architecture to use")
    args = parser.parse_args()

    #Data Splitting
    graph_directory = args.graph_dir
    seed = 51
    train_ratio = 0.8
    #Model Values
    hidden_dim = 256
    output_dim = 2
    #Training Values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 30

    #Preparing Data
    data_list = prepare_data(graph_directory)
    loader_train, loader_test = prepare_splits(data_list, seed, train_ratio, batch_size)
    num_node_features = data_list[0].x.size(1)  # Get the number of features from the first graph

    #Starting Model
    model = Node_GIN(num_node_features, hidden_dim, output_dim).to(device)

    if args.gnn_model == 1:
        model = Node_GCN(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 2:
        model = Node_GIN(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 3:
        model = Node_GCN_Embed(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 4:
        model = Node_GIN_Embed(num_node_features, hidden_dim, output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 10], device=device))

    #Training
    print(f'Number of node features: {num_node_features}')
    print(f'Using device: {device}')
    print('Starting training cycles.')
    train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm = train_loop(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm)
    print('Final Epoch AUC:', auc)
    print('Final Epoch F1 Score:', f1)

    torch.save({
        'model_state_dict': model.state_dict()
    }, f'saved_models/{args.gnn_model}.pt')

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

