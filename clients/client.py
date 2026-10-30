"""Simulated Independent Industrial Client for Federated Learning.

This module simulates industrial organizations or facilities:
- Client 1: Factory A (Automotive Manufacturing Plant)
- Client 2: Factory B (Electronics Assembly Facility)
- Client 3: Factory C (Chemical / SCADA Substation)

CRITICAL PRIVACY PRINCIPLE:
Each client maintains its own private dataset. Raw network traffic data NEVER leaves
the client object. Only model parameters (weights) and cryptographic update hashes
are shared with the Federated Learning aggregator.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import IIoTIntrusionMLP


class IndustrialClient:
    """Independent Industrial Client node participating in Collaborative Intrusion Detection.

    Maintains local model weights, private dataset, and conducts local training.
    """

    def __init__(
        self,
        client_id: str,
        display_name: str,
        input_dim: int = 15,
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_seed: int = 42,
    ):
        self.client_id = client_id
        self.display_name = display_name
        self.input_dim = input_dim
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize local PyTorch Neural Network
        torch.manual_seed(random_seed)
        self.model = IIoTIntrusionMLP(input_dim=input_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Private local dataset containers (kept strictly within client)
        self.train_loader: Optional[DataLoader] = None
        self.sample_count: int = 0
        self.normal_count: int = 0
        self.attack_count: int = 0

    def set_local_data(self, X_local: np.ndarray, y_local: np.ndarray) -> None:
        """Load and encapsulate private industrial training traffic.

        Raw network records are stored strictly in memory inside this client instance.
        """
        self.sample_count = len(y_local)
        self.normal_count = int(np.sum(y_local == 0))
        self.attack_count = int(np.sum(y_local == 1))

        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(X_local, dtype=torch.float32)
        y_tensor = torch.tensor(y_local, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_local_model(
        self,
        epochs: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Perform local training on private dataset.

        Returns a dictionary of execution metadata and model parameters.
        Does NOT return raw training data.
        """
        if self.train_loader is None or self.sample_count == 0:
            raise ValueError(f"[{self.client_id}] Cannot train: local dataset is empty!")

        num_epochs = epochs if epochs is not None else self.local_epochs
        self.model.train()

        start_time = time.time()
        epoch_losses: List[float] = []

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            total_batches = 0

            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            avg_epoch_loss = running_loss / max(total_batches, 1)
            epoch_losses.append(avg_epoch_loss)

            if verbose:
                print(f"  [{self.client_id}] Epoch {epoch}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

        training_time = time.time() - start_time
        final_loss = epoch_losses[-1] if epoch_losses else 0.0

        # Evaluate accuracy on local training data
        local_acc = self._compute_local_accuracy()

        # Compute cryptographic hash of current local model parameters
        model_hash = self.model.compute_hash()

        return {
            "client_id": self.client_id,
            "display_name": self.display_name,
            "sample_count": self.sample_count,
            "normal_count": self.normal_count,
            "attack_count": self.attack_count,
            "epochs_completed": num_epochs,
            "final_loss": round(final_loss, 4),
            "epoch_losses": epoch_losses,
            "local_accuracy": round(local_acc * 100, 2),
            "training_time_sec": round(training_time, 4),
            "model_hash": model_hash,
            "weights": self.model.get_weights(),
        }

    def _compute_local_accuracy(self) -> float:
        """Internal helper to calculate training accuracy on local data."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                preds = torch.argmax(self.model(X_batch), dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        return correct / total if total > 0 else 0.0

    def evaluate(self, X_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
        """Evaluate client's current model on an external or global test dataset.

        Computes standard intrusion detection metrics:
        - Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR)
        """
        self.model.eval()
        X_tensor = torch.tensor(X_eval, dtype=torch.float32)

        with torch.no_grad():
            preds = self.model.predict(X_tensor)
            probs = self.model.predict_proba(X_tensor)

        acc = accuracy_score(y_eval, preds)
        prec = precision_score(y_eval, preds, zero_division=0)
        rec = recall_score(y_eval, preds, zero_division=0)
        f1 = f1_score(y_eval, preds, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_eval, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr * 100, 2),
            "confusion_matrix": cm.tolist(),
            "predictions": preds,
            "probabilities": probs,
        }

    def set_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Update local model with new weights received from the global aggregator."""
        self.model.set_weights(weights)

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Export local model weights for federated aggregation."""
        return self.model.get_weights()

    def get_model_hash(self) -> str:
        """Return the SHA-256 hash of the client's current model parameters."""
        return self.model.compute_hash()


if __name__ == "__main__":
    from preprocessing.data_preprocessing import run_preprocessing_pipeline

    print("=" * 70)
    print("STAGE 3: INDEPENDENT LOCAL MODEL TRAINING TEST")
    print("=" * 70)

    # 1. Load preprocessed data and partitions
    data_res = run_preprocessing_pipeline(generate_plots=False, use_non_iid=False)
    partitions = data_res["client_partitions"]
    X_test = data_res["X_test"]
    y_test = data_res["y_test"]

    client_configs = [
        ("Client_1_Factory_A", "Factory A (Automotive)"),
        ("Client_2_Factory_B", "Factory B (Electronics)"),
        ("Client_3_Factory_C", "Factory C (SCADA Substation)"),
    ]

    clients: List[IndustrialClient] = []

    # 2. Instantiate and load private data into each client
    for cid, dname in client_configs:
        client = IndustrialClient(client_id=cid, display_name=dname, input_dim=15, local_epochs=5)
        pdata = partitions[cid]
        client.set_local_data(pdata["X"], pdata["y"])
        clients.append(client)
        print(f"Initialized {dname}: {client.sample_count} samples (Normal: {client.normal_count}, Attack: {client.attack_count})")

    # 3. Train each client independently
    print("\n--- BEGINNING INDEPENDENT LOCAL TRAINING ---")
    local_reports = []
    for client in clients:
        report = client.train_local_model(epochs=5, verbose=False)
        local_reports.append(report)
        print(f"[{client.display_name}]")
        print(f"  Training Time:    {report['training_time_sec']:.3f} s")
        print(f"  Final Loss:       {report['final_loss']:.4f}")
        print(f"  Local Accuracy:   {report['local_accuracy']:.2f}%")
        print(f"  Model Hash (SHA): {report['model_hash'][:16]}...")

    # 4. Evaluate each isolated local model on the common test set
    print("\n--- ISOLATED LOCAL MODELS TEST EVALUATION ---")
    for client in clients:
        eval_res = client.evaluate(X_test, y_test)
        print(f"[{client.display_name} on Held-Out Test Set]")
        print(f"  Accuracy:  {eval_res['accuracy']:.2f}%")
        print(f"  Precision: {eval_res['precision']:.2f}%")
        print(f"  Recall:    {eval_res['recall']:.2f}%")
        print(f"  F1 Score:  {eval_res['f1_score']:.4f}")
        print(f"  FPR:       {eval_res['false_positive_rate']:.2f}%")

    print("\n[SUCCESS] Stage 3 Independent Local Training Verified!")
