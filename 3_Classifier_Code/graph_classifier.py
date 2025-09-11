import argparse
import gc
import glob
import json
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

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
import umap

def get_graph(graph, target):
    with open(graph, 'rb') as f:
        g_x, g_edges, weights, node_ids, x_metric, g_metric = pickle.load(f)

    x = torch.tensor(g_x, dtype=torch.float)             #Shape = [num_nodes, num_features]
    edge_index = torch.tensor(g_edges, dtype=torch.long) #Shape = [2, num_edges]
    edge_attr = torch.tensor(weights, dtype=torch.float) #Shape = [num_edges]
    node_ids = torch.tensor(node_ids, dtype=torch.long)  #Shape = [num_nodes]
    y = torch.tensor(target, dtype=torch.float)          #Value = 0/1
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, node_ids=node_ids, y=y)
    return data

def prepare_data(graphs_dir, target_json):
    pattern = os.path.join(graphs_dir, "*.pkl")
    file_list = sorted(glob.glob(pattern))

    with open(target_json, 'r') as f:
        target_dict = json.load(f)

    data_list = []
    for file in file_list:
        strain = os.path.basename(file).split(".")[0]
        data = get_graph(file, target=target_dict[strain])
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

    #Just checking number of graphs with MDR+ or - in both lists.
    train_labels = torch.tensor([data.y.item() for data in train_list], dtype=torch.long)
    test_labels = torch.tensor([data.y.item() for data in test_list], dtype=torch.long)
    print("Train label counts:", torch.bincount(train_labels))
    print("Test label counts:", torch.bincount(test_labels))

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

def calculate_accuracy(predictions, targets):
    preds = predictions.argmax(dim=1)  # PyTorch version of argmax
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

#Training process (gradient accumulation to make use of GPU).
def train(model, loader_train, loader_test, optimizer, criterion, scheduler, device):
    loss_train = 0
    loss_test = 0
    acc_test = 0
    acc_train = 0

    model.train()
    torch.cuda.empty_cache()
    gc.collect()

    for data in loader_train:
        optimizer.zero_grad()

        data = data.to(device)
        out = model(data)
        # print(out.shape)
        # print( "ngraphs", data.num_graphs, len(data) )

        # Binary classification, so target is of shape [batch_size, 1]
        loss = criterion(out, data.y.long())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        with torch.no_grad():
            acc_train += calculate_accuracy(out, data.y)

        del data

    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.long())
            loss_test += loss.item()
            acc_test += calculate_accuracy(out, data.y)

            probs = torch.softmax(out, dim=1)[:, 1]  # Probability of class 1 (MDR+)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(data.y.cpu())

        scheduler.step(loss_test)

    avg_train_loss = loss_train / len(loader_train)
    avg_train_acc = acc_train / len(loader_train)
    avg_test_loss = loss_test / len(loader_test)
    avg_test_acc = acc_test / len(loader_test)

    auc, prauc, f1, cm = calculate_metrics(all_preds, all_probs, all_labels)
    print(f"AUC: {auc:.3f}, PRAUC: {prauc:.3f}, F1: {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    return avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc, auc, f1, cm

#Training Loop
def train_model(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs):
    #Train the model for a number of epochs and return the loss and accuracy history.
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        loss_train, acc_train, loss_test, acc_test, auc, f1, cm = train(
            model, loader_train, loader_test, optimizer, criterion, scheduler, device
        )
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        print(f'Epoch: {epoch}, Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.4f}, ' +
              f'Test Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}')
        #Option to view embedding at set epochs
        #if epoch % 10 == 0:
        #    view_embedding(model, loader_test, device, epoch)

    return train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm


def plot_results(train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm):
    #Plot loss, accuracy, and final AUC/F1 score into a single multi-page PDF.
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

def view_embedding(model, test_loader, device, epoch=30):
    # Collect pooled graph embeddings
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            node_ids = data.node_ids
            embed = model.embedding(node_ids)  # [num_nodes, emb_dim]
            pooled = global_mean_pool(embed, data.batch)  # [num_graphs, emb_dim]
            embeddings.append(pooled.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    # Combine all embeddings and labels
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # UMAP projection
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title('UMAP Projection of Graph Embeddings')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.colorbar(label='MDR Status (0=–, 1=+)')
    plt.tight_layout()
    plt.savefig(f'embedding_umap_epoch{epoch}.png', dpi=300)
    plt.close()


class GNNClassifier_GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super(GNNClassifier_GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr

        # weighted GCN layers
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # global pooling and final classification
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

class GNNClassifier_GCN_embed(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.1):
        super(GNNClassifier_GCN_embed, self).__init__()
        self.embedding = nn.Embedding(55049, embedding_dim)
        self.conv1 = GCNConv(num_node_features + embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr
        node_ids = data.node_ids

        # concatenate node features + node_id embeddings
        embed = self.embedding(node_ids)
        x = torch.cat([x, embed], dim=1)

        # weighted GCN layers
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # global pooling and final classification
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

class GNNClassifier_GIN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.05):
        super().__init__()

        # 2-layer MLP for GINConv
        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.LeakyReLU(0.1),
                nn.Linear(out_dim, out_dim, bias=True),
                nn.LeakyReLU(0.1)
            )

        # GIN layers
        self.conv1 = GINConv(make_mlp(num_node_features, hidden_dim), train_eps=False)
        self.conv2 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)
        self.conv3 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)
        self.conv4 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_ids = data.node_ids

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GNNClassifier_GIN_embed(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, embedding_dim=1024, dropout=0.05):
        super().__init__()

        self.embedding = nn.Embedding(55049, embedding_dim)

        # 2-layer MLP for GINConv
        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.LeakyReLU(0.1),
                nn.Linear(out_dim, out_dim, bias=True),
                nn.LeakyReLU(0.1)
            )

        # GIN layers
        self.conv1 = GINConv(make_mlp(num_node_features + embedding_dim, hidden_dim), train_eps=False)
        self.conv2 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)
        self.conv3 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)
        self.conv4 = GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=False)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_ids = data.node_ids

        embed = self.embedding(node_ids) * 3
        x = torch.cat([x, embed], dim=1)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gnn_model", type=int, default=1, help="Which GNN architecture to use")
    args = parser.parse_args()

    #Data Splitting
    graph_directory = "graphs_final/"
    target_directory = 'dataframes/mdr1000.json'
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
    data_list = prepare_data(graph_directory, target_directory)
    loader_train, loader_test = prepare_splits(data_list, seed, train_ratio, batch_size)
    num_node_features = data_list[0].x.size(1)  # Get the number of features from the first graph

    #Starting Model
    if args.gnn_model == 1:
        model = GNNClassifier_GCN(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 2:
        model = GNNClassifier_GIN(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 3:
        model = GNNClassifier_GCN_embed(num_node_features, hidden_dim, output_dim).to(device)
    elif args.gnn_model == 4:
        model = GNNClassifier_GIN_embed(num_node_features, hidden_dim, output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.66, 2.08], device=device))

    #Training
    print(f'Number of node features: {num_node_features}')
    print(f'Using device: {device}')
    print('Starting training cycles.')
    train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm = train_model(model, loader_train, loader_test, optimizer, criterion, scheduler, device, epochs)
    plot_results(train_losses, test_losses, train_accuracies, test_accuracies, auc, f1, cm)
    print('Final Epoch AUC:', auc)
    print('Final Epoch F1 Score:', f1)

    MODEL_NAMES = {
        1: "GNNClassifier_GCN",
        2: "GNNClassifier_GIN",
        3: "GNNClassifier_GCN_embed",
        4: "GNNClassifier_GIN_embed",
    }

    # Save the trained model state
    torch.save(model.state_dict(),
               f"saved_models/{MODEL_NAMES[args.gnn_model]}.pt")
    print('Model Saved')

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
