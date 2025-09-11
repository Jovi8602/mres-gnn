import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
from graph_classifier import (prepare_data, prepare_splits,GNNClassifier_GCN, GNNClassifier_GIN,GNNClassifier_GCN_embed, GNNClassifier_GIN_embed,)

def main():
    # Config
    graph_dir   = "graphs_final/"
    target_file = "dataframes/mdr1000.json"
    seed        = 51
    train_ratio = 0.8
    batch_size  = 32
    hidden_dim  = 256
    output_dim  = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    data_list = prepare_data(graph_dir, target_file)
    data_list = prepare_data(graph_dir, target_file)
    _, loader_test = prepare_splits(data_list, seed=seed, train_ratio=train_ratio, batch_size=batch_size)
    num_node_features = data_list[0].x.size(1)

    # --- Model registry ---
    MODEL_REG = {
        "GNNClassifier_GCN":        GNNClassifier_GCN,
        "GNNClassifier_GIN":        GNNClassifier_GIN,
        "GNNClassifier_GCN_embed":  GNNClassifier_GCN_embed,
        "GNNClassifier_GIN_embed":  GNNClassifier_GIN_embed,
    }

    # #Inference
    all_results = {}

    for name, cls in MODEL_REG.items():
        print(f"\n=== Evaluating {name} ===")
        model_path = f"saved_models/{name}.pt"

        # Build + load
        model = cls(num_node_features, hidden_dim, output_dim).to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()

        outputs, targets = [], []
        with torch.no_grad():
            for data in loader_test:
                data = data.to(device)
                logits = model(data)
                probs = F.softmax(logits, dim=1)[:, 1]    # MDR+ prob
                outputs.extend(probs.cpu().numpy())
                targets.extend(data.y.cpu().numpy())

        all_results[name] = {
            "y_target": np.array(targets),
            "y_probs": np.array(outputs),
        }

    #Plot ROC Curve. Requires data from code above.
    # Define model names and their corresponding variables
    mlp_results = joblib.load("saved_models/mlp_results.pkl")
    rf_results = joblib.load("saved_models/rf_results.pkl")
    models = {
        'MLP': (mlp_results['y_target'], mlp_results['y_probs']),
        'RF': (rf_results['y_target'], rf_results['y_probs']),
        'GCN': (all_results['GNNClassifier_GCN']['y_target'], all_results['GNNClassifier_GCN']['y_probs']),
        'GCN + Embedding': (all_results['GNNClassifier_GCN_embed']['y_target'], all_results['GNNClassifier_GCN_embed']['y_probs']),
        'GIN': (all_results['GNNClassifier_GIN']['y_target'], all_results['GNNClassifier_GIN']['y_probs']),
        'GIN + Embedding': (all_results['GNNClassifier_GIN_embed']['y_target'], all_results['GNNClassifier_GIN_embed']['y_probs'])
    }

    #Plot and Save
    plt.figure(figsize=(8, 6))
    for model_name, (targets, outputs) in models.items():
        fpr, tpr, _ = roc_curve(targets, outputs)
        auc = roc_auc_score(targets, outputs)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill (AUC = 0.50)') #Baseline
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for All Classifiers', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("roc_comparison_all_models.png", dpi=300)
    print('ROC Figure Saved')

if __name__ == "__main__":
    main()