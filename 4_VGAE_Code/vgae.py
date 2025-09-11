import argparse
import gc
import glob
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj

from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score)

def get_graph(graph):
    with open(graph, 'rb') as f:
        g_x, g_edges, weights, node_ids, x_metric, g_metric = pickle.load(f)

    x = torch.tensor(g_x, dtype=torch.float)             #Shape = [num_nodes, num_features]
    edge_index = torch.tensor(g_edges, dtype=torch.long) #Shape = [2, num_edges]
    edge_attr = torch.tensor(weights, dtype=torch.float) #Shape = [num_edges]
    #node_ids = torch.tensor(node_ids, dtype=torch.long)  #Shape = [num_nodes]
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
    return data

def prepare_data(graphs_dir):
    pattern = os.path.join(graphs_dir, "*.pkl")
    file_list = sorted(glob.glob(pattern))

    data_list = []
    check = False
    for file in file_list:
        data = get_graph(file)
        data_list.append(data)
        if not check:
            print('Node Data Shape:', data.x)
            print('Edge Index Shape:', data.edge_index)
            print('Edge Attributes/Weights:', data.edge_attr, '\n')
            check = True

    print(f'Total number of graphs: {len(data_list)}')

    return data_list

def prepare_splits(data_list, seed, train_ratio, batch_size):
    np.random.seed(seed)
    np.random.shuffle(data_list)
    split_idx = int(train_ratio * len(data_list))
    train_list = data_list[:split_idx]
    test_list = data_list[split_idx:]

    print(f'Number of training graphs: {len(train_list)}')
    print(f'Number of test graphs: {len(test_list)} \n')

    loader_train = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_list, batch_size=batch_size, shuffle=True)
    print("Number of training batches:", len(loader_train))
    print("Total graphs in loader_train:", len(loader_train.dataset))

    return loader_train, loader_test

def generate_indices_batch(batch):
    num_graphs = batch.max() + 1  # Number of graphs in batch
    # print("num_graphs:", num_graphs)
    edge_indices = []

    for graph_id in range(num_graphs):
        nodes = (batch == graph_id).nonzero(as_tuple=True)[0]
        n = nodes.size(0)

        if n == 0:
            continue

        row = nodes.repeat(n)
        col = nodes.unsqueeze(1).repeat(1, n).flatten()
        edge_indices.append(torch.stack([row, col]))

    return torch.cat(edge_indices, dim=1)

def reconstruction_loss(recon_edge_index, recon_edge_weights, edge_index, mu, logvar, batch):
    #Converts to full adjacency matrix
    adj_true = to_dense_adj(edge_index, batch=batch)
    adj_pred = to_dense_adj(recon_edge_index, batch=batch, edge_attr=recon_edge_weights)

    # Squeezing probabilities away from exactly 0 and 1 to prevent calculation errors.
    eps = 1e-7
    adj_pred = adj_pred.clamp(eps, 1-eps)

    #Calculating weights, and loss using weights.
    pos_weight = (adj_true == 0).sum().float() / (adj_true == 1).sum().float()
    weight = adj_true * pos_weight + (1 - adj_true)

    loss = F.binary_cross_entropy(adj_pred.view(-1), adj_true.view(-1), weight=weight.view(-1))

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss += 0.1*kl

    return loss

def calculate_metrics(recon_edge_index, recon_edge_weights, edge_index, batch):
    device = recon_edge_weights.device
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Dense true and predicted adjacency
    adj_true = to_dense_adj(edge_index, batch=batch).to(device)         # [num_graphs, N, N]
    adj_pred = to_dense_adj(recon_edge_index, batch=batch, edge_attr=recon_edge_weights).to(device)
    adj_pred_binary = (adj_pred > 0.9).float()

    f1_scores = []
    prec_scores = []
    rec_scores = []

    for i in range(adj_true.size(0)):
        y_true = adj_true[i].view(-1).cpu().numpy()
        y_pred = adj_pred_binary[i].view(-1).cpu().numpy()
        #Flattens both tensors to be 1-dimensional.

        if y_true.sum() == 0 and y_pred.sum() == 0:
            # Perfect prediction on empty graph
            f1_scores.append(1.0)
            prec_scores.append(1.0)
            rec_scores.append(1.0)
        else:
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            prec_scores.append(precision_score(y_true, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_true, y_pred, zero_division=0))

    # Compute average over graphs
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_prec = sum(prec_scores) / len(prec_scores)
    avg_rec = sum(rec_scores) / len(rec_scores)

    return avg_f1, avg_prec, avg_rec

def inspect_predictions(model, data_loader, device, num_batches=1):
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(data_loader):
            if batch_id >= num_batches:
                break

            data = data.to(device)
            mu, logvar = model(data)
            z = model.reparameterize(mu, logvar)

            recon_edge_index = generate_indices_batch(data.batch).to(device)
            recon_edge_weights = model.decode(z, recon_edge_index, data.batch)

            print(f"\n--- Batch {batch_id} ---")
            print(f"Number of predicted edges: {recon_edge_index.size(1)}")
            print(f"Predicted edge probabilities (min/max/mean): {recon_edge_weights.min().item():.4f} / "
                  f"{recon_edge_weights.max().item():.4f} / "
                  f"{recon_edge_weights.mean().item():.4f}")

            # Binarize
            binary_preds = (recon_edge_weights > 0.9).float()
            num_predicted_edges = binary_preds.sum().item()
            print(f"Number of predicted edges > 0.9: {int(num_predicted_edges)}")

            true_adj = to_dense_adj(data.edge_index, batch=data.batch).cpu()
            pred_adj = to_dense_adj(recon_edge_index, batch=data.batch, edge_attr=binary_preds).cpu()

            # Compare a few graphs
            for i in range(min(3, true_adj.size(0))):
                print(f"\nGraph {i}:")
                print(f"Number of nodes: {true_adj[i].size(0)}")
                print(f"True edges: {(true_adj[i] > 0).sum().item()}")
                print(f"Predicted edges: {(pred_adj[i] > 0).sum().item()}")

                correct_preds = ((true_adj[i].bool()) & (pred_adj[i].bool())).sum().item()
                print(f"Correctly predicted edges: {correct_preds}")

def train(model, loader_train, loader_test, optimizer, criterion, scheduler, device):
    loss_train, total_f1, total_prec, total_rec = 0, 0, 0, 0

    model.train()
    torch.cuda.empty_cache()
    gc.collect()

    graph_num = 0
    for data in loader_train:
        data = data.to(device)
        optimizer.zero_grad()

        # Encode batch (compute mu and logvar)
        mu, logvar = model(data)  # Pass batch tensor
        z = model.reparameterize(mu, logvar)
        recon_edge_index = generate_indices_batch(data.batch).to(device)
        recon_edge_weights = model.decode(z, recon_edge_index, data.batch)

        print(' \nBATCH NUMBER:', graph_num)
        print('reconstructed adjacency shape:', recon_edge_index.size())
        print('reconstructed edge weights sample:', recon_edge_weights[0:2])
        graph_num += 1

        loss = reconstruction_loss(recon_edge_index, recon_edge_weights, data.edge_index, mu, logvar, data.batch)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            f1, prec, rec = calculate_metrics(recon_edge_index, recon_edge_weights, data.edge_index, data.batch)
            total_f1 += f1
            total_prec += prec
            total_rec += rec

        del data
        model.train()

    num_batch = len(loader_train)
    avg_train_loss = loss_train / num_batch
    f1 = total_f1 / num_batch
    rec = total_rec / num_batch
    prec = total_prec / num_batch
    train_metrics = [avg_train_loss, f1, prec, rec]

    model.eval()
    loss_test, total_f1, total_prec, total_rec = 0, 0, 0, 0

    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)

            mu, logvar = model(data)  # Pass batch tensor
            z = model.reparameterize(mu, logvar)
            recon_edge_index = generate_indices_batch(data.batch).to(device)
            recon_edge_weights = model.decode(z, recon_edge_index, data.batch)

            loss = reconstruction_loss(recon_edge_index, recon_edge_weights, data.edge_index, mu, logvar, data.batch)
            loss_test += loss.item()

            f1, prec, rec = calculate_metrics(recon_edge_index, recon_edge_weights, data.edge_index, data.batch)
            total_f1 += f1
            total_prec += prec
            total_rec += rec

        if math.isfinite(loss_test):
            scheduler.step(loss_test)

    num_batch = len(loader_test)
    avg_test_loss = loss_test / num_batch
    f1 = total_f1 / num_batch
    rec = total_rec / num_batch
    prec = total_prec / num_batch
    test_metrics = [avg_test_loss, f1, prec, rec]

    return train_metrics, test_metrics

#Training Loop
def train_loop(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Initial predictions on test set:")
    inspect_predictions(model, loader_test, device, num_batches=1)

    for epoch in range(1, epochs + 1):
        train_metrics, test_metrics = train(
            model, loader_train, loader_test, optimizer, criterion, scheduler, device
        )
        train_losses.append(train_metrics[0])
        test_losses.append(test_metrics[0])
        train_accuracies.append(train_metrics[1])
        test_accuracies.append(test_metrics[1])
        print(f'Epoch: {epoch}, Train Loss: {train_metrics[0]:.4f}, Train Accuracy: {train_metrics[1]:.4f}, ' +
              f'Test Loss: {test_metrics[0]:.4f}, Test Accuracy: {test_metrics[1]:.4f}')
        print(f"[Full Test Metrics]\nF1: {test_metrics[1]:.3f}, Recall: {test_metrics[2]:.3f}, Precision: {test_metrics[3]:.3f}")

    print("Final predictions on test set:")
    inspect_predictions(model, loader_test, device, num_batches=1)

    return train_losses, test_losses, train_accuracies, test_accuracies


def plot_results(train_losses, test_losses, train_accuracies, test_accuracies):
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
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train F1')
        plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test F1')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy over Epochs')
        plt.legend()
        pdf.savefig()
        plt.close()

class GIN_Encoder(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, latent_dim):
        super().__init__()

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim, bias=True),
                nn.ReLU(),
                nn.LayerNorm(out_dim, eps=1e-5),
            )

        self.conv1 = GINConv(make_mlp(num_node_features, hidden_dim), train_eps=False)
        self.conv2 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)
        self.conv3 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)

        self.lin1 = nn.Linear(hidden_dim, latent_dim)
        self.lin2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = F.relu(x)
        mu = self.lin1(x)
        logvar = self.lin2(x)

        return mu, logvar

class VGAEModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, latent_dim):
        super(VGAEModel, self).__init__()

        self.encoder = GIN_Encoder(num_node_features, hidden_dim, latent_dim)

        #Trainable parameters for decoder.
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.zeros(1))

        print(self.w.device)

    def forward(self, data):
        mu, logvar = self.encoder(data)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    #Choose decoder model here
    def decode(self, z, edge_index, batch=None):
        return self.decode_bilinear(z, edge_index, batch)

    def decode_bilinear(self, z, edge_index, batch=None):
        src, tgt = edge_index[0], edge_index[1]
        z_src, z_tgt = z[src], z[tgt]             # both shape [E, D]

        # element‐wise multiplication, then weight by self.w, sum dims
        scores = (z_src * z_tgt * self.w).sum(dim=-1) + self.b
        probs  = torch.sigmoid(scores)

        if batch is not None:
            mask = batch[src] == batch[tgt]
            edge_scores = probs * mask.float()

        return edge_scores

    def decode_dot(self, z, edge_index, batch=None):
        src, tgt = edge_index
        z_src, z_tgt = z[src], z[tgt]
        scores = (z_src * z_tgt).sum(dim=-1)
        probs = torch.sigmoid(scores)
        if batch is not None:
            mask = batch[src] == batch[tgt]
            edge_scores = probs * mask.float()

        return edge_scores

def main():
    #Data Splitting
    graph_directory = "graphs_final/"
    seed = 51
    train_ratio = 0.8
    #Model Values
    hidden_dim = 256
    latent_dim = 64
    #Training Values
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    epochs = 50

    #Preparing Data
    data_list = prepare_data(graph_directory)
    loader_train, loader_test = prepare_splits(data_list, seed, train_ratio, batch_size)
    num_node_features = data_list[0].x.size(1)  # Get the number of features from the first graph

    #Starting Model
    model = VGAEModel(num_node_features, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = 'Blank'

    #Training
    print(f'Number of node features: {num_node_features}')
    print(f'Using device: {device}')
    print('Starting training cycles.')
    train_losses, test_losses, train_accuracies, test_accuracies = train_loop(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies)

    torch.save({
        'model_state_dict': model.state_dict()
    }, 'saved_models/vgae_saved.pth')
    print("✅ Saved model status to vgae_saved.pth")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
