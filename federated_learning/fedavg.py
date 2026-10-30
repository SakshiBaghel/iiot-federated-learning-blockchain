"""Federated Learning Server and Federated Averaging (FedAvg) Module.

This module implements the core Federated Learning coordination pipeline:
1. Initialize a global neural network model.
2. Broadcast global model parameters to all simulated industrial clients (Factory A, B, C).
3. Coordinate independent local training on private client datasets.
4. Collect parameter updates and cryptographic hashes (NO raw data is transferred).
5. Aggregate updates using sample-size-weighted Federated Averaging (FedAvg):
       W_global = SUM( (n_k / N_total) * W_k )
6. Update the global model and evaluate on the separate held-out test dataset.
7. Record round-by-round convergence history (loss, accuracy, F1, FPR).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clients.client import IndustrialClient
from models.model import IIoTIntrusionMLP


def federated_averaging(
    client_updates: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Perform Federated Averaging (FedAvg) on client model state_dict parameters.

    Mathematical Formulation:
        Let K be the number of participating clients.
        Let n_k be the number of private training samples on client k.
        Let N = SUM(n_k) be the total training samples across all clients.
        Let W_k be the parameter dictionary (weights & biases) of client k.

        The aggregated global weight tensor is:
            W_global = SUM_{k=1}^K ( (n_k / N) * W_k )

    Args:
        client_updates: List of client report dicts containing 'sample_count' and 'weights'.

    Returns:
        Aggregated state_dict dictionary with averaged parameters.
    """
    if not client_updates:
        raise ValueError("Cannot perform FedAvg on an empty list of client updates.")

    total_samples = sum(u["sample_count"] for u in client_updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be greater than zero.")

    # Reference keys from the first client's state_dict
    ref_keys = list(client_updates[0]["weights"].keys())
    aggregated_weights: Dict[str, torch.Tensor] = {}

    for key in ref_keys:
        # Initialize zero tensor matching shape and float dtype
        ref_tensor = client_updates[0]["weights"][key]
        accumulated = torch.zeros_like(ref_tensor, dtype=torch.float32)

        # Accumulate weighted parameters from each client
        for update in client_updates:
            n_k = update["sample_count"]
            weight_ratio = n_k / total_samples
            client_tensor = update["weights"][key].to(torch.float32)
            accumulated += client_tensor * weight_ratio

        # Convert back to matching dtype of reference
        aggregated_weights[key] = accumulated.to(ref_tensor.dtype)

    return aggregated_weights


class FederatedServer:
    """Central Federated Learning Orchestration Server.

    Manages global rounds, broadcasts models, invokes FedAvg, and tracks metrics.
    """

    def __init__(
        self,
        clients: List[IndustrialClient],
        input_dim: int = 15,
        random_seed: int = 42,
    ):
        self.clients = clients
        self.input_dim = input_dim

        # Initialize Global Neural Network
        torch.manual_seed(random_seed)
        self.global_model = IIoTIntrusionMLP(input_dim=input_dim)
        self.criterion = nn.CrossEntropyLoss()

        self.round_history: List[Dict[str, Any]] = []

    def broadcast_global_weights(self) -> None:
        """Broadcast current global model weights to all participating clients."""
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_model_weights(global_weights)

    def evaluate_global_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate current global model on the held-out test dataset."""
        self.global_model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            logits = self.global_model(X_tensor)
            loss = self.criterion(logits, torch.tensor(y_test, dtype=torch.long)).item()
            preds = self.global_model.predict(X_tensor)
            probs = self.global_model.predict_proba(X_tensor)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        # Confusion Matrix & False Positive Rate (FPR)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "loss": round(loss, 4),
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr * 100, 2),
            "confusion_matrix": cm.tolist(),
            "predictions": preds,
            "probabilities": probs,
        }

    def train_round(
        self,
        round_number: int,
        local_epochs: int = 3,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        on_round_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute one complete Federated Learning communication round.

        Pipeline for each round:
        1. Record pre-aggregation global model hash.
        2. Broadcast global model weights to all clients.
        3. Each client conducts independent local training.
        4. Collect client weights and cryptographic hashes.
        5. Aggregate weights using FedAvg.
        6. Update global model with aggregated weights.
        7. Compute post-aggregation global model hash.
        8. Evaluate updated global model on held-out test set.
        9. Trigger optional blockchain recording hook.
        """
        round_start_time = time.time()
        initial_global_hash = self.global_model.compute_hash()

        if verbose:
            print(f"\n" + "=" * 60)
            print(f"COMMUNICATION ROUND {round_number} START")
            print(f"=" * 60)
            print(f"[ROUND {round_number}] Initial Global Model Hash: {initial_global_hash[:16]}...")
            print(f"[ROUND {round_number}] Step 1: Broadcasting global model to {len(self.clients)} clients...")

        # 1. Broadcast global weights to all clients
        self.broadcast_global_weights()

        # 2. Local Training on each client
        client_updates: List[Dict[str, Any]] = []
        client_hashes: Dict[str, str] = {}
        total_training_time = 0.0

        if verbose:
            print(f"[ROUND {round_number}] Step 2: Clients training locally on private datasets...")

        for client in self.clients:
            update = client.train_local_model(epochs=local_epochs, verbose=False)
            client_updates.append(update)
            client_hashes[client.client_id] = update["model_hash"]
            total_training_time += update["training_time_sec"]

            if verbose:
                print(
                    f"  -> [{client.display_name}] {update['sample_count']} samples | "
                    f"Loss: {update['final_loss']:.4f} | Local Acc: {update['local_accuracy']:.2f}% | "
                    f"Hash: {update['model_hash'][:12]}..."
                )

        # 3. Federated Averaging (FedAvg) Parameter Aggregation
        if verbose:
            print(f"[ROUND {round_number}] Step 3: Server aggregating weights via FedAvg...")

        aggregation_start = time.time()
        new_global_weights = federated_averaging(client_updates)
        self.global_model.set_weights(new_global_weights)
        aggregation_time = time.time() - aggregation_start

        updated_global_hash = self.global_model.compute_hash()
        round_total_time = time.time() - round_start_time

        if verbose:
            print(f"[ROUND {round_number}] Updated Global Model Hash: {updated_global_hash[:16]}...")

        # 4. Evaluation on Held-Out Test Set
        eval_metrics = {}
        if X_test is not None and y_test is not None:
            eval_metrics = self.evaluate_global_model(X_test, y_test)
            if verbose:
                print(f"[ROUND {round_number}] Step 4: Global Model Test Evaluation:")
                print(
                    f"  -> Accuracy: {eval_metrics['accuracy']:.2f}% | "
                    f"Precision: {eval_metrics['precision']:.2f}% | "
                    f"Recall: {eval_metrics['recall']:.2f}% | "
                    f"F1: {eval_metrics['f1_score']:.4f} | "
                    f"FPR: {eval_metrics['false_positive_rate']:.2f}%"
                )

        # Average client loss for this round
        avg_client_loss = float(np.mean([u["final_loss"] for u in client_updates]))

        round_record = {
            "round": round_number,
            "timestamp": time.time(),
            "initial_global_hash": initial_global_hash,
            "updated_global_hash": updated_global_hash,
            "client_hashes": client_hashes,
            "participating_clients": [c.client_id for c in self.clients],
            "client_samples": {c.client_id: c.sample_count for c in self.clients},
            "avg_client_loss": round(avg_client_loss, 4),
            "client_updates": client_updates,
            "eval_metrics": eval_metrics,
            "training_time_sec": round(total_training_time, 4),
            "aggregation_time_sec": round(aggregation_time, 6),
            "round_duration_sec": round(round_total_time, 4),
        }

        self.round_history.append(round_record)

        # Trigger external hook (e.g. blockchain recording in Phase 6)
        if on_round_complete is not None:
            on_round_complete(round_record)

        return round_record

    def run_federated_learning(
        self,
        num_rounds: int = 5,
        local_epochs: int = 3,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        on_round_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run complete Federated Learning pipeline across multiple communication rounds."""
        if verbose:
            print("\n" + "=" * 70)
            print(f"STARTING FEDERATED LEARNING: {num_rounds} ROUNDS, {len(self.clients)} CLIENTS")
            print("=" * 70)

        for r in range(1, num_rounds + 1):
            self.train_round(
                round_number=r,
                local_epochs=local_epochs,
                X_test=X_test,
                y_test=y_test,
                on_round_complete=on_round_complete,
                verbose=verbose,
            )

        if verbose:
            print("\n" + "=" * 70)
            print("FEDERATED LEARNING PIPELINE COMPLETED")
            print("=" * 70)

        return self.round_history

    def plot_convergence_curves(self, output_dir: Path) -> List[str]:
        """Generate plots of FedAvg performance across communication rounds."""
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_plots = []

        rounds = [r["round"] for r in self.round_history]
        accuracies = [r["eval_metrics"].get("accuracy", 0.0) for r in self.round_history]
        losses = [r["avg_client_loss"] for r in self.round_history]
        f1_scores = [r["eval_metrics"].get("f1_score", 0.0) for r in self.round_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

        # Plot 1: Global Test Accuracy & F1-Score
        ax1.plot(rounds, accuracies, marker="o", color="#2980b9", linewidth=2.5, label="Test Accuracy (%)")
        ax1.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Global Model Accuracy over FedAvg Rounds", fontsize=12, fontweight="bold")
        ax1.set_xticks(rounds)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_ylim([max(0, min(accuracies) - 5), 105])
        ax1.legend(loc="lower right")

        # Plot 2: Average Client Training Loss
        ax2.plot(rounds, losses, marker="s", color="#e74c3c", linewidth=2.5, label="Avg Client Loss")
        ax2.set_xlabel("Communication Round", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Training Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
        ax2.set_title("Training Loss Convergence over FedAvg Rounds", fontsize=12, fontweight="bold")
        ax2.set_xticks(rounds)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

        plt.suptitle("Federated Learning (FedAvg) Convergence Dynamics", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        p1 = output_dir / "fedavg_convergence_curves.png"
        plt.savefig(p1, bbox_inches="tight")
        plt.close()
        generated_plots.append(str(p1))

        return generated_plots


if __name__ == "__main__":
    from preprocessing.data_preprocessing import run_preprocessing_pipeline

    print("=" * 70)
    print("STAGE 4: FEDERATED LEARNING (FedAvg) TEST EXECUTION")
    print("=" * 70)

    # 1. Load data and partitions
    data_res = run_preprocessing_pipeline(generate_plots=False, use_non_iid=False)
    partitions = data_res["client_partitions"]
    X_test = data_res["X_test"]
    y_test = data_res["y_test"]

    # 2. Setup Clients
    client_configs = [
        ("Client_1_Factory_A", "Factory A (Automotive)"),
        ("Client_2_Factory_B", "Factory B (Electronics)"),
        ("Client_3_Factory_C", "Factory C (SCADA Substation)"),
    ]

    clients = []
    for cid, dname in client_configs:
        c = IndustrialClient(client_id=cid, display_name=dname, input_dim=15, local_epochs=3)
        c.set_local_data(partitions[cid]["X"], partitions[cid]["y"])
        clients.append(c)

    # 3. Setup Server and Run 5 Rounds
    server = FederatedServer(clients=clients, input_dim=15)
    history = server.run_federated_learning(
        num_rounds=5,
        local_epochs=3,
        X_test=X_test,
        y_test=y_test,
        verbose=True,
    )

    # 4. Generate Convergence Graphs
    graphs_dir = PROJECT_ROOT / "results" / "graphs"
    plot_files = server.plot_convergence_curves(graphs_dir)
    print(f"\nGenerated Convergence Plots: {plot_files}")

    print("\n[SUCCESS] Stage 4 Federated Learning (FedAvg) Verified!")
