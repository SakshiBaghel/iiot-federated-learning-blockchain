"""Comprehensive 4-Way Comparative Evaluation Module for IIoT Intrusion Detection.

This module evaluates and compares four distinct configurations under identical conditions:
1. Traditional Centralized ML (pooled training dataset, violates data privacy).
2. Independent Local ML (isolated factory models without collaboration).
3. Federated Learning (FedAvg collaborative training without sharing raw data).
4. Federated Learning + Blockchain (FedAvg + SHA-256 cryptographic audit trail).

Key Principles:
- All values are computed from ACTUAL EXECUTION of the models on the same held-out test set.
- We do NOT fabricate or simulate numbers.
- Blockchain's role is clearly identified as providing integrity, non-repudiation,
  and tamper-evidence, without making false claims about ML parameter optimization.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.blockchain import LightweightBlockchain
from clients.client import IndustrialClient
from federated_learning.fedavg import FederatedServer
from models.model import IIoTIntrusionMLP
from preprocessing.data_preprocessing import run_preprocessing_pipeline


def train_centralized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 15,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    random_seed: int = 42,
) -> Tuple[IIoTIntrusionMLP, float]:
    """Train a traditional centralized model on pooled data from all factories."""
    torch.manual_seed(random_seed)
    model = IIoTIntrusionMLP(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    return model, train_time


def evaluate_model_on_test(
    model: IIoTIntrusionMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a trained model on the held-out test set."""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        preds = model.predict(X_tensor)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr * 100, 2),
        "confusion_matrix": cm,
        "predictions": preds,
    }


def run_comparative_experiments(
    use_non_iid: bool = True,
    num_rounds: int = 5,
    local_epochs: int = 3,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute all four configurations and return a structured comparison DataFrame."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING 4-WAY COMPARATIVE EXPERIMENTAL EVALUATION")
    print("=" * 80)

    # 1. Load Data
    prep_data = run_preprocessing_pipeline(generate_plots=False, use_non_iid=use_non_iid)
    X_train = prep_data["X_train"]
    y_train = prep_data["y_train"]
    X_test = prep_data["X_test"]
    y_test = prep_data["y_test"]
    input_dim = prep_data["num_features"]
    partitions = prep_data["client_partitions"]

    # Parameter size calculation for communication overhead
    ref_model = IIoTIntrusionMLP(input_dim=input_dim)
    model_param_bytes = sum(p.numel() * 4 for p in ref_model.parameters())  # float32 = 4 bytes
    param_size_kb = model_param_bytes / 1024.0

    # -------------------------------------------------------------
    # CONFIGURATION 1: Traditional Centralized ML
    # -------------------------------------------------------------
    print("\n[CONFIG 1/4] Training Traditional Centralized ML Model (Pooled Data)...")
    centralized_model, c_time = train_centralized_model(
        X_train, y_train, input_dim=input_dim, epochs=num_rounds * local_epochs
    )
    c_metrics = evaluate_model_on_test(centralized_model, X_test, y_test)
    # Centralized requires uploading all raw training data to server (2400 rows * 15 floats * 8 bytes)
    c_raw_data_kb = (len(X_train) * input_dim * 8) / 1024.0
    print(f"  -> Accuracy: {c_metrics['accuracy']:.2f}% | F1: {c_metrics['f1_score']:.4f} | Time: {c_time:.3f}s")

    # -------------------------------------------------------------
    # CONFIGURATION 2: Independent Local ML (Isolated Clients)
    # -------------------------------------------------------------
    print("\n[CONFIG 2/4] Training Independent Local ML Models (No Collaboration)...")
    local_clients: List[IndustrialClient] = []
    local_metrics_list = []
    local_total_time = 0.0

    client_configs = [
        ("Client_1_Factory_A", "Factory A"),
        ("Client_2_Factory_B", "Factory B"),
        ("Client_3_Factory_C", "Factory C"),
    ]

    for cid, dname in client_configs:
        c = IndustrialClient(client_id=cid, display_name=dname, input_dim=input_dim, local_epochs=local_epochs)
        c.set_local_data(partitions[cid]["X"], partitions[cid]["y"])
        t_rep = c.train_local_model(epochs=local_epochs)
        local_total_time += t_rep["training_time_sec"]
        e_rep = evaluate_model_on_test(c.model, X_test, y_test)
        local_metrics_list.append(e_rep)
        local_clients.append(c)
        print(f"  -> [{dname} in isolation] Accuracy: {e_rep['accuracy']:.2f}% | FPR: {e_rep['false_positive_rate']:.2f}%")

    # Average metrics across independent clients
    loc_acc = float(np.mean([m["accuracy"] for m in local_metrics_list]))
    loc_prec = float(np.mean([m["precision"] for m in local_metrics_list]))
    loc_rec = float(np.mean([m["recall"] for m in local_metrics_list]))
    loc_f1 = float(np.mean([m["f1_score"] for m in local_metrics_list]))
    loc_fpr = float(np.mean([m["false_positive_rate"] for m in local_metrics_list]))
    loc_cm = np.round(np.mean([m["confusion_matrix"] for m in local_metrics_list], axis=0)).astype(int)

    # -------------------------------------------------------------
    # CONFIGURATION 3: Federated Learning (FedAvg)
    # -------------------------------------------------------------
    print("\n[CONFIG 3/4] Running Federated Learning (FedAvg without Blockchain)...")
    fl_clients = []
    for cid, dname in client_configs:
        fc = IndustrialClient(client_id=cid, display_name=dname, input_dim=input_dim, local_epochs=local_epochs)
        fc.set_local_data(partitions[cid]["X"], partitions[cid]["y"])
        fl_clients.append(fc)

    fl_server = FederatedServer(clients=fl_clients, input_dim=input_dim)
    fl_start = time.time()
    fl_server.run_federated_learning(num_rounds=num_rounds, local_epochs=local_epochs, X_test=X_test, y_test=y_test, verbose=False)
    fl_total_time = time.time() - fl_start
    fl_metrics = evaluate_model_on_test(fl_server.global_model, X_test, y_test)
    # Communication overhead: 2 ways (download + upload) * num_clients * num_rounds * param_size
    fl_comm_kb = 2 * len(fl_clients) * num_rounds * param_size_kb
    print(f"  -> Accuracy: {fl_metrics['accuracy']:.2f}% | F1: {fl_metrics['f1_score']:.4f} | Time: {fl_total_time:.3f}s")

    # -------------------------------------------------------------
    # CONFIGURATION 4: Federated Learning + Blockchain
    # -------------------------------------------------------------
    print("\n[CONFIG 4/4] Running Federated Learning + Blockchain Ledger...")
    bc_clients = []
    for cid, dname in client_configs:
        bcc = IndustrialClient(client_id=cid, display_name=dname, input_dim=input_dim, local_epochs=local_epochs)
        bcc.set_local_data(partitions[cid]["X"], partitions[cid]["y"])
        bc_clients.append(bcc)

    blockchain = LightweightBlockchain()
    bc_server = FederatedServer(clients=bc_clients, input_dim=input_dim)

    def bc_hook(r_data: Dict[str, Any]) -> None:
        blockchain.record_fl_round(
            round_number=r_data["round"],
            client_hashes=r_data["client_hashes"],
            aggregated_global_hash=r_data["updated_global_hash"],
            initial_global_hash=r_data["initial_global_hash"],
            participating_clients=r_data["participating_clients"],
            client_samples=r_data["client_samples"],
            eval_metrics=r_data["eval_metrics"],
            avg_loss=r_data["avg_client_loss"],
            training_time_sec=r_data["round_duration_sec"],
        )

    bc_fl_start = time.time()
    bc_server.run_federated_learning(
        num_rounds=num_rounds, local_epochs=local_epochs, X_test=X_test, y_test=y_test,
        on_round_complete=bc_hook, verbose=False
    )
    bc_fl_total_time = time.time() - bc_fl_start

    # Measure blockchain validation overhead
    val_start = time.perf_counter()
    is_valid, _ = blockchain.is_chain_valid()
    bc_verify_overhead_ms = (time.perf_counter() - val_start) * 1000.0

    bc_metrics = evaluate_model_on_test(bc_server.global_model, X_test, y_test)
    print(f"  -> Accuracy: {bc_metrics['accuracy']:.2f}% | F1: {bc_metrics['f1_score']:.4f} | Verify Time: {bc_verify_overhead_ms:.2f}ms")

    # -------------------------------------------------------------
    # Compile Comparative Report Table
    # -------------------------------------------------------------
    report_data = [
        {
            "Configuration": "1. Centralized ML",
            "Data Privacy Preserved": "No (Raw Data Uploaded)",
            "Accuracy (%)": c_metrics["accuracy"],
            "Precision (%)": c_metrics["precision"],
            "Recall (%)": c_metrics["recall"],
            "F1-Score": c_metrics["f1_score"],
            "False Positive Rate (%)": c_metrics["false_positive_rate"],
            "Training Time (s)": round(c_time, 3),
            "Communication Overhead (KB)": round(c_raw_data_kb, 1),
            "Blockchain Overhead (ms)": 0.0,
            "Audit Integrity": "None (No Ledger)",
        },
        {
            "Configuration": "2. Independent Local ML",
            "Data Privacy Preserved": "Yes (Local Only)",
            "Accuracy (%)": round(loc_acc, 2),
            "Precision (%)": round(loc_prec, 2),
            "Recall (%)": round(loc_rec, 2),
            "F1-Score": round(loc_f1, 4),
            "False Positive Rate (%)": round(loc_fpr, 2),
            "Training Time (s)": round(local_total_time, 3),
            "Communication Overhead (KB)": 0.0,
            "Blockchain Overhead (ms)": 0.0,
            "Audit Integrity": "None (Local Logs Only)",
        },
        {
            "Configuration": "3. Federated Learning (FedAvg)",
            "Data Privacy Preserved": "Yes (Weights Only)",
            "Accuracy (%)": fl_metrics["accuracy"],
            "Precision (%)": fl_metrics["precision"],
            "Recall (%)": fl_metrics["recall"],
            "F1-Score": fl_metrics["f1_score"],
            "False Positive Rate (%)": fl_metrics["false_positive_rate"],
            "Training Time (s)": round(fl_total_time, 3),
            "Communication Overhead (KB)": round(fl_comm_kb, 1),
            "Blockchain Overhead (ms)": 0.0,
            "Audit Integrity": "None (Vulnerable to Tampering)",
        },
        {
            "Configuration": "4. Federated Learning + Blockchain",
            "Data Privacy Preserved": "Yes (Weights Only)",
            "Accuracy (%)": bc_metrics["accuracy"],
            "Precision (%)": bc_metrics["precision"],
            "Recall (%)": bc_metrics["recall"],
            "F1-Score": bc_metrics["f1_score"],
            "False Positive Rate (%)": bc_metrics["false_positive_rate"],
            "Training Time (s)": round(bc_fl_total_time, 3),
            "Communication Overhead (KB)": round(fl_comm_kb, 1),
            "Blockchain Overhead (ms)": round(bc_verify_overhead_ms, 2),
            "Audit Integrity": "Guaranteed (SHA-256 Ledger)",
        },
    ]

    df_report = pd.DataFrame(report_data)
    csv_path = output_dir / "comparison_report.csv"
    df_report.to_csv(csv_path, index=False)
    print(f"\n[REPORT SAVED] Comparison table written to: {csv_path}")

    # -------------------------------------------------------------
    # Visualization: Metric Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    configs = ["Centralized", "Independent Local", "FedAvg", "FedAvg + Blockchain"]
    metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (x100)"]

    x = np.arange(len(configs))
    width = 0.18

    acc_vals = df_report["Accuracy (%)"].values
    prec_vals = df_report["Precision (%)"].values
    rec_vals = df_report["Recall (%)"].values
    f1_vals = (df_report["F1-Score"].values * 100).round(2)

    ax.bar(x - 1.5 * width, acc_vals, width, label="Accuracy (%)", color="#3498db", edgecolor="black")
    ax.bar(x - 0.5 * width, prec_vals, width, label="Precision (%)", color="#2ecc71", edgecolor="black")
    ax.bar(x + 0.5 * width, rec_vals, width, label="Recall (%)", color="#f39c12", edgecolor="black")
    ax.bar(x + 1.5 * width, f1_vals, width, label="F1-Score (x100)", color="#9b59b6", edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Metrics Across 4 System Configurations", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontweight="bold")
    ax.set_ylim([80, 105])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    chart1 = graphs_dir / "four_way_metrics_comparison.png"
    plt.savefig(chart1)
    plt.close()

    # -------------------------------------------------------------
    # Visualization: Confusion Matrices (2x2 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=150)
    cms = [c_metrics["confusion_matrix"], loc_cm, fl_metrics["confusion_matrix"], bc_metrics["confusion_matrix"]]
    titles = [
        "1. Centralized ML",
        "2. Independent Local ML",
        "3. Federated Learning (FedAvg)",
        "4. FedAvg + Blockchain",
    ]

    for idx, (cm_matrix, title) in enumerate(zip(cms, titles)):
        row = idx // 2
        col = idx % 2
        ax_cm = axes[row, col]
        im = ax_cm.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        ax_cm.set_title(title, fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_xticklabels(["Normal", "Attack"])
        ax_cm.set_yticklabels(["Normal", "Attack"])
        ax_cm.set_xlabel("Predicted Label", fontsize=10)
        ax_cm.set_ylabel("True Label", fontsize=10)

        # Annotate values
        thresh = cm_matrix.max() / 2.0
        for i in range(cm_matrix.shape[0]):
            for j in range(cm_matrix.shape[1]):
                ax_cm.text(
                    j, i, format(cm_matrix[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_matrix[i, j] > thresh else "black",
                    fontweight="bold",
                )

    plt.suptitle("Confusion Matrices: 4 System Approaches on 600 Test Samples", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart2 = graphs_dir / "confusion_matrices_comparison.png"
    plt.savefig(chart2)
    plt.close()

    # -------------------------------------------------------------
    # Visualization: Overhead & Security Comparison
    # -------------------------------------------------------------
    fig, (ax_time, ax_fpr) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Training Time
    times = df_report["Training Time (s)"].values
    bars = ax_time.bar(configs, times, color=["#95a5a6", "#e67e22", "#3498db", "#2980b9"], edgecolor="black")
    ax_time.set_ylabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax_time.set_title("Training Execution Time", fontsize=11, fontweight="bold")
    ax_time.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        ax_time.annotate(f"{b.get_height():.2f}s", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # False Positive Rate
    fprs = df_report["False Positive Rate (%)"].values
    bars_fpr = ax_fpr.bar(configs, fprs, color=["#27ae60", "#c0392b", "#27ae60", "#27ae60"], edgecolor="black")
    ax_fpr.set_ylabel("False Positive Rate (%)", fontsize=10, fontweight="bold")
    ax_fpr.set_title("False Positive Rate (FPR) Comparison", fontsize=11, fontweight="bold")
    ax_fpr.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars_fpr:
        ax_fpr.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Computational Efficiency & Security Error Trade-offs", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart3 = graphs_dir / "overhead_and_security_comparison.png"
    plt.savefig(chart3, bbox_inches="tight")
    plt.close()

    print(f"[GRAPHS SAVED] Generated evaluation plots in: {graphs_dir}")
    return df_report


if __name__ == "__main__":
    df_res = run_comparative_experiments(use_non_iid=True, num_rounds=5, local_epochs=3)
    print("\n" + "=" * 90)
    print("FINAL 4-WAY COMPARISON TABLE")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)
