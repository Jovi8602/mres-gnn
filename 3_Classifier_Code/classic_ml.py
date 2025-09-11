#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
import shap

presence_data = pd.read_csv("dataframes/presence.csv")
metadata = pd.read_csv("dataframes/metadata.csv")

#Creating info_list (abs/pres) and mdr_list for basic NN training.
random_ints = random.sample(range(1, 5001), 5000)
has_absence = presence_data.columns
strains = [has_absence[i] for i in random_ints]

info_list, mdr_list = [], []
check, index = 0, 0
while check != 1000:
    s = strains[index]
    if s.upper() in metadata['ID'].tolist():
        info = presence_data[s].tolist()
        del info[0]
        mdr = metadata.loc[metadata['ID'] == s.upper(), 'MDR'].item()
        if mdr == 'Yes':
            info_list.append(info); mdr_list.append(1); check += 1
        elif mdr == 'No':
            info_list.append(info); mdr_list.append(0); check += 1
    index += 1

print('Number of strains:', len(mdr_list), len(info_list))
print('Number of genes per strain:', len(info_list[0]))

# Dataset + Model classes
class PresenceDataset(Dataset):
    def __init__(self, info_list, mdr_list):
        self.X = torch.tensor(info_list, dtype=torch.float32)
        self.y = torch.tensor(mdr_list, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim=55049, hidden_dim=256, dropout=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

def calculate_accuracy(outputs, targets):
    preds = (torch.sigmoid(outputs) >= 0.5).float()
    return (preds == targets).float().mean().item()

# Train/Test split
num_samples = 1000
input_dim = len(info_list[0])

indices = np.arange(num_samples)
np.random.shuffle(indices)
split_idx = int(0.8 * num_samples)
train_indices, test_indices = indices[:split_idx], indices[split_idx:]

train_info = [info_list[i] for i in train_indices]
train_mdr  = [mdr_list[i] for i in train_indices]
test_info  = [info_list[i] for i in test_indices]
test_mdr   = [mdr_list[i] for i in test_indices]

train_dataset = PresenceDataset(train_info, train_mdr)
test_dataset  = PresenceDataset(test_info, test_mdr)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the MLP
model = SimpleNN(input_dim=input_dim, hidden_dim=256, dropout=0.5).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses, test_losses, train_accs, test_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, running_acc, total_train = 0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * Xb.size(0)
        running_acc  += calculate_accuracy(outputs, yb) * Xb.size(0)
        total_train  += Xb.size(0)
    train_losses.append(running_loss / total_train)
    train_accs.append(running_acc / total_train)

    model.eval()
    running_loss, running_acc, total_test = 0, 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            running_loss += loss.item() * Xb.size(0)
            running_acc  += calculate_accuracy(outputs, yb) * Xb.size(0)
            total_test   += Xb.size(0)
    test_losses.append(running_loss / total_test)
    test_accs.append(running_acc / total_test)

# Final evaluation
model.eval()
all_outputs, all_targets = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        outputs = model(Xb)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_outputs.extend(probs)
        all_targets.extend(yb.numpy())
mlp_output = np.array(all_outputs).flatten()
mlp_target = np.array(all_targets).flatten()

pred_labels = (mlp_output >= 0.5).astype(int)
auc   = roc_auc_score(mlp_target, mlp_output)
prauc = average_precision_score(mlp_target, mlp_output)
f1    = f1_score(mlp_target, pred_labels)
cm    = confusion_matrix(mlp_target, pred_labels)

# Save PDF (loss, accuracy, metrics)
with PdfPages("mlp_report.pdf") as pdf:
    # Loss curve
    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    ax1.plot(range(1, num_epochs+1), test_losses,  label="Test Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs"); ax1.legend()
    pdf.savefig(fig1); plt.close(fig1)

    # Accuracy curve
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.plot(range(1, num_epochs+1), train_accs, label="Train Acc")
    ax2.plot(range(1, num_epochs+1), test_accs,  label="Test Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs"); ax2.legend()
    pdf.savefig(fig2); plt.close(fig2)

    # Final metrics page
    summary = (
        "MLP Final Metrics (Test Set)\n"
        "────────────────────────────\n"
        f"AUC   : {auc:.4f}\n"
        f"PRAUC : {prauc:.4f}\n"
        f"F1    : {f1:.4f}\n"
        f"Confusion Matrix:\n{cm}"
    )
    fig3 = plt.figure(figsize=(8.5,5))
    fig3.text(0.02, 0.98, summary, va="top", family="monospace", size=12)
    pdf.savefig(fig3); plt.close(fig3)

print("Saved mlp_report.pdf")

# Save y_true, y_probs
joblib.dump({"y_target": mlp_target, "y_probs": mlp_output}, "saved_models/mlp_results.pkl")
print("Saved mlp_results.pkl")

# SHAP feature importance
X_train_tensor = torch.tensor(train_info, dtype=torch.float32)
X_test_tensor  = torch.tensor(test_info,  dtype=torch.float32)
model_cpu = model.to("cpu").eval()

explainer = shap.GradientExplainer(model_cpu, X_train_tensor[:100])
shap_values = explainer.shap_values(X_test_tensor)

def to_2d_meanabs(sv):
    if isinstance(sv, list):
        arr = np.array(sv[1]) if len(sv) == 2 else np.mean(np.abs(np.stack(sv, axis=0)), axis=0)
    else:
        arr = np.array(sv)
    while arr.ndim > 2:
        arr = arr.mean(axis=-1)
    return arr

sv_2d = to_2d_meanabs(shap_values)
importance = np.mean(np.abs(sv_2d), axis=0)

all_genes = list(presence_data['Strain'])[1:]  # drop header
top_k = 20
idx = np.argsort(importance)[-top_k:][::-1]
top_names = [all_genes[i] for i in idx]
top_vals  = importance[idx]

plt.figure(figsize=(10, 6))
plt.barh(range(top_k), top_vals)
plt.yticks(range(top_k), top_names)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top Contributing Genes (MLP)")
plt.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig("mlp_top_genes.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.close()

print("Saved mlp_top_genes_bar.png")

# Train Random Forest
X = np.array(info_list)
y = np.array(mdr_list)

num_samples = len(y)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split_idx = int(0.8 * num_samples)

train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions + metrics
y_pred  = rf.predict(X_test)
y_probs = rf.predict_proba(X_test)[:, 1]

rf_acc   = accuracy_score(y_test, y_pred)
rf_auc   = roc_auc_score(y_test, y_probs)
rf_prauc = average_precision_score(y_test, y_probs)
rf_f1    = f1_score(y_test, y_pred)
rf_cm    = confusion_matrix(y_test, y_pred)

# Append metrics page to PDF
with PdfPages("rf_report.pdf") as pdf:

    summary_rf = (
        "Random Forest Metrics (Test Set)\n"
        "─────────────────────────────────\n"
        f"Accuracy : {rf_acc:.4f}\n"
        f"AUC      : {rf_auc:.4f}\n"
        f"PRAUC    : {rf_prauc:.4f}\n"
        f"F1 Score : {rf_f1:.4f}\n"
        f"Confusion Matrix:\n{rf_cm}"
    )

    fig_rf = plt.figure(figsize=(8.5,5))
    fig_rf.text(0.02, 0.98, summary_rf, va="top", family="monospace", size=12)
    pdf.savefig(fig_rf)   # assumes you already have `with PdfPages(...) as pdf:`
    plt.close(fig_rf)

# Save y_true, y_probs
joblib.dump({"y_target": y_test, "y_probs": y_probs}, "saved_models/rf_results.pkl")
print("Saved rf_results.pkl")

# Feature importance plot
all_genes = list(presence_data['Strain'])[1:]
importances = rf.feature_importances_

top_k = 20
indices = np.argsort(importances)[-top_k:][::-1]
top_names = [all_genes[i] for i in indices]
top_vals  = importances[indices]

plt.figure(figsize=(10, 6))
plt.barh(range(top_k), top_vals)
plt.yticks(range(top_k), top_names)
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top Contributing Genes (Random Forest)")
plt.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig("rf_top_genes.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.close()

print("Saved rf_top_genes_bar.png")
