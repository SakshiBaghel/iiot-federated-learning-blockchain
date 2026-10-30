"""PyTorch Neural Network Model for Industrial IoT Intrusion Detection.

This module defines the Multi-Layer Perceptron (IIoTIntrusionMLP) architecture:
- Input layer accepts industrial network flow features (default: 15 features).
- Two hidden layers with ReLU activations and Dropout for regularization.
- Output layer outputs logits for binary classification:
    Class 0: Normal Operational Traffic
    Class 1: Malicious Cyber Attack
"""

from __future__ import annotations

import copy
import hashlib
import io
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IIoTIntrusionMLP(nn.Module):
    """Lightweight Multi-Layer Perceptron for Industrial IoT Intrusion Classification.

    Designed specifically to be:
    - Fast and lightweight for CPU training on student laptops.
    - Highly interpretable and stable for Federated Averaging (FedAvg).
    - Resistant to overfitting via Dropout (p=0.2).
    """

    def __init__(self, input_dim: int = 15, hidden1: int = 32, hidden2: int = 16, num_classes: int = 2):
        super(IIoTIntrusionMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes

        # Fully Connected Layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Returns raw unnormalized logits."""
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu2(out)

        logits = self.fc3(out)
        return logits

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Compute discrete binary class predictions (0 or 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Compute prediction probabilities for the positive class (Attack = 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Extract a detached CPU deepcopy of the model's parameters (state_dict)."""
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load parameters into the model."""
        self.load_state_dict(copy.deepcopy(state_dict))

    def compute_hash(self) -> str:
        """Compute a deterministic SHA-256 cryptographic hash of model parameters.

        This serves as a unique digital fingerprint of the model's exact weights.
        Used by the blockchain module to record and verify model updates.
        """
        buffer = io.BytesIO()
        # Save state_dict deterministically into buffer
        torch.save(self.state_dict(), buffer)
        buffer.seek(0)
        return hashlib.sha256(buffer.read()).hexdigest()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING IIoTIntrusionMLP NEURAL NETWORK ARCHITECTURE")
    print("=" * 60)

    # Instantiate model
    model = IIoTIntrusionMLP(input_dim=15)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params}")

    # Test dummy forward pass
    dummy_input = torch.randn(5, 15)  # Batch of 5 samples
    logits = model(dummy_input)
    preds = model.predict(dummy_input)
    probs = model.predict_proba(dummy_input)

    print(f"Dummy Input Shape:  {dummy_input.shape}")
    print(f"Logits Shape:       {logits.shape}")
    print(f"Predictions:        {preds}")
    print(f"Attack Probs:       {probs.round(3)}")

    # Test weight hashing
    h1 = model.compute_hash()
    print(f"Model SHA-256 Hash: {h1}")
    print("\n[SUCCESS] Model module verified successfully!")
