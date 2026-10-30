"""Lightweight Simulated Blockchain Ledger for Federated Learning Auditability.

This module provides a pure-Python, cryptographic blockchain ledger using SHA-256:
1. Block creation with deterministic SHA-256 hashing.
2. Strict cryptographic linking: each block stores 'previous_hash'.
3. Immutable logging of Federated Learning communication rounds:
   - Round number
   - Timestamp
   - Participating Client IDs (Factory A, Factory B, Factory C)
   - Cryptographic SHA-256 hashes of client model updates
   - Aggregated Global Model SHA-256 hash
   - Training samples and evaluation metrics
4. Blockchain integrity validation (detects broken hashes or severed links).
5. Controlled tampering simulation for college demonstration.

CRITICAL PRIVACY PRINCIPLE:
Raw industrial network traffic is NEVER stored on the blockchain.
Only metadata, summary metrics, and cryptographic hashes (digital fingerprints) are stored.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Block:
    """Represents a single block in the Federated Learning audit ledger."""

    def __init__(
        self,
        index: int,
        timestamp: float,
        data: Dict[str, Any],
        previous_hash: str,
        nonce: int = 0,
    ):
        self.index = index
        self.timestamp = timestamp
        self.data = copy.deepcopy(data)
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate deterministic SHA-256 hash of all block contents.

        Uses sorted JSON keys so identical content always yields the identical hash.
        """
        block_content = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        encoded = json.dumps(block_content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to readable dictionary."""
        return {
            "index": self.index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "timestamp_unix": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "nonce": self.nonce,
        }


class LightweightBlockchain:
    """Lightweight, tamper-evident Python blockchain for auditing Federated Learning."""

    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self) -> Block:
        """Create the foundational Genesis Block (Block #0)."""
        genesis_data = {
            "event": "GENESIS_BLOCK",
            "message": "Blockchain-Enabled Secure Federated Learning for Industrial IoT Ledger",
            "protocol_version": "1.0-academic",
            "consensus": "Proof-of-Authority / Federated Consensus Simulation",
            "created_by": "IIoT Security Consortium (Factories A, B, C)",
        }
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            data=genesis_data,
            previous_hash="0" * 64,  # Standard 64 zeros for Genesis
        )
        self.chain.append(genesis_block)
        return genesis_block

    @property
    def last_block(self) -> Block:
        """Return the most recently added block in the chain."""
        return self.chain[-1]

    def record_fl_round(
        self,
        round_number: int,
        client_hashes: Dict[str, str],
        aggregated_global_hash: str,
        initial_global_hash: str,
        participating_clients: List[str],
        client_samples: Dict[str, int],
        eval_metrics: Dict[str, Any],
        avg_loss: float,
        training_time_sec: float,
    ) -> Block:
        """Record a completed Federated Learning round as a new block on the ledger.

        Args:
            round_number: Communication round index.
            client_hashes: Mapping of client IDs to their model update SHA-256 hashes.
            aggregated_global_hash: SHA-256 hash of the resulting global model.
            initial_global_hash: Pre-round global model hash before FedAvg aggregation.
            participating_clients: List of client IDs that submitted updates.
            client_samples: Number of training samples per client.
            eval_metrics: Global model evaluation results (accuracy, precision, recall, F1).
            avg_loss: Average local training loss across clients.
            training_time_sec: Total round training and aggregation time.

        Returns:
            The newly created and linked Block.
        """
        round_transaction = {
            "event": "FEDERATED_LEARNING_ROUND_RECORD",
            "round_number": round_number,
            "participating_clients": participating_clients,
            "client_samples": client_samples,
            "client_model_hashes": client_hashes,
            "initial_global_model_hash": initial_global_hash,
            "aggregated_global_model_hash": aggregated_global_hash,
            "avg_client_loss": avg_loss,
            "test_accuracy": eval_metrics.get("accuracy", 0.0),
            "test_f1_score": eval_metrics.get("f1_score", 0.0),
            "test_precision": eval_metrics.get("precision", 0.0),
            "test_recall": eval_metrics.get("recall", 0.0),
            "false_positive_rate": eval_metrics.get("false_positive_rate", 0.0),
            "round_duration_sec": training_time_sec,
        }

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=round_transaction,
            previous_hash=self.last_block.hash,
        )

        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self) -> Tuple[bool, str]:
        """Verify the integrity and consistency of the entire blockchain.

        Checks:
        1. That each block's recorded hash matches its recalculated SHA-256 hash.
        2. That each block's 'previous_hash' matches the actual hash of the preceding block.

        Returns:
            (True, "All blocks verified successfully") or
            (False, "Description of tampering detected at Block X")
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check 1: Has block content been altered?
            expected_hash = current.calculate_hash()
            if current.hash != expected_hash:
                return (
                    False,
                    f"TAMPERING DETECTED at Block #{current.index}! "
                    f"Recorded Hash: {current.hash[:16]}... != Recalculated Hash: {expected_hash[:16]}...",
                )

            # Check 2: Has the chain link been severed?
            if current.previous_hash != previous.hash:
                return (
                    False,
                    f"BROKEN CHAIN LINK between Block #{previous.index} and Block #{current.index}! "
                    f"Current's PrevHash: {current.previous_hash[:16]}... != Previous's Hash: {previous.hash[:16]}...",
                )

        return True, "Blockchain is valid: All cryptographic hashes and chain links are verified."

    def verify_update_hash(self, round_number: int, client_id: str, candidate_hash: str) -> Tuple[bool, str]:
        """Verify whether a candidate model update hash matches what is recorded on-chain."""
        for block in self.chain:
            if block.data.get("round_number") == round_number:
                recorded_hashes = block.data.get("client_model_hashes", {})
                recorded_hash = recorded_hashes.get(client_id)

                if recorded_hash is None:
                    return False, f"No record found for client '{client_id}' in Round #{round_number}."

                if recorded_hash == candidate_hash:
                    return True, f"SUCCESS: Client '{client_id}' update hash matches on-chain record for Round #{round_number}."
                else:
                    return False, f"FRAUD DETECTED: Hash mismatch for '{client_id}' in Round #{round_number}!"

        return False, f"Round #{round_number} block not found on the blockchain."

    def simulate_tampering_attack(
        self,
        block_index: int = 1,
        fake_hash: str = "bad000000000000000000000000000000000000000000000000000000000cafe",
    ) -> Dict[str, Any]:
        """Simulate an attacker attempting to modify a recorded client hash in an older block.

        Demonstrates why blockchains are tamper-evident:
        Changing even a single character in a past block causes hash verification to fail!
        """
        if block_index >= len(self.chain) or block_index == 0:
            raise ValueError("Select a valid non-genesis block index to tamper with.")

        target_block = self.chain[block_index]
        original_data = copy.deepcopy(target_block.data)
        original_hash = target_block.hash

        # Attacker secretly alters the recorded model update hash of Factory A
        if "client_model_hashes" in target_block.data:
            first_client = list(target_block.data["client_model_hashes"].keys())[0]
            target_block.data["client_model_hashes"][first_client] = fake_hash
        else:
            target_block.data["tampered_entry"] = "MALICIOUS_MODIFICATION"

        # Check blockchain validity post-tampering
        is_valid_after_tamper, message = self.is_chain_valid()

        # Restore original data so our demonstration does not permanently corrupt the chain
        target_block.data = original_data
        target_block.hash = original_hash

        return {
            "tampered_block_index": block_index,
            "original_hash": original_hash,
            "detection_message": message,
            "tamper_detected": not is_valid_after_tamper,
        }

    def export_chain_json(self) -> str:
        """Export the full ledger as formatted JSON for visualization and reporting."""
        return json.dumps([b.to_dict() for b in self.chain], indent=2)


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 5: LIGHTWEIGHT BLOCKCHAIN LEDGER TEST")
    print("=" * 70)

    # 1. Initialize Blockchain
    bc = LightweightBlockchain()
    print(f"Initialized Blockchain with Genesis Block:")
    print(f"  Index:         {bc.chain[0].index}")
    print(f"  Hash:          {bc.chain[0].hash}")
    print(f"  Previous Hash: {bc.chain[0].previous_hash}")

    # 2. Record simulated Federated Learning Round 1
    sample_client_hashes = {
        "Client_1_Factory_A": "40861ea0678b02b1123456789abcdef0123456789abcdef0123456789abcdef0",
        "Client_2_Factory_B": "763a0d7fe78f3222123456789abcdef0123456789abcdef0123456789abcdef0",
        "Client_3_Factory_C": "6a4939089d677890123456789abcdef0123456789abcdef0123456789abcdef0",
    }

    block1 = bc.record_fl_round(
        round_number=1,
        client_hashes=sample_client_hashes,
        aggregated_global_hash="0ac6b1f383f5427e123456789abcdef0123456789abcdef0123456789abcdef0",
        initial_global_hash="a6975553ca1f8550123456789abcdef0123456789abcdef0123456789abcdef0",
        participating_clients=list(sample_client_hashes.keys()),
        client_samples={"Client_1_Factory_A": 800, "Client_2_Factory_B": 800, "Client_3_Factory_C": 800},
        eval_metrics={"accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1_score": 1.0, "false_positive_rate": 0.0},
        avg_loss=0.015,
        training_time_sec=0.45,
    )
    print(f"\nAdded Block #1 (FL Round 1):")
    print(f"  Hash:          {block1.hash}")
    print(f"  Previous Hash: {block1.previous_hash}")
    print(f"  Linked correctly? {block1.previous_hash == bc.chain[0].hash}")

    # 3. Add Round 2
    block2 = bc.record_fl_round(
        round_number=2,
        client_hashes=sample_client_hashes,
        aggregated_global_hash="5e6b00122c0ee58c123456789abcdef0123456789abcdef0123456789abcdef0",
        initial_global_hash=block1.hash,
        participating_clients=list(sample_client_hashes.keys()),
        client_samples={"Client_1_Factory_A": 800, "Client_2_Factory_B": 800, "Client_3_Factory_C": 800},
        eval_metrics={"accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1_score": 1.0, "false_positive_rate": 0.0},
        avg_loss=0.005,
        training_time_sec=0.42,
    )
    print(f"\nAdded Block #2 (FL Round 2):")
    print(f"  Hash:          {block2.hash}")
    print(f"  Previous Hash: {block2.previous_hash}")
    print(f"  Linked correctly? {block2.previous_hash == block1.hash}")

    # 4. Verify legitimate blockchain
    print("\n--- CRYPTOGRAPHIC CHAIN INTEGRITY CHECK ---")
    valid, msg = bc.is_chain_valid()
    print(f"Is Blockchain Valid? {valid}")
    print(f"Status Message:      {msg}")

    # 5. Tampering Demonstration
    print("\n--- CONTROLLED TAMPERING ATTACK SIMULATION ---")
    print("Scenario: An attacker modifies Factory A's recorded model hash in Block #1.")
    tamper_result = bc.simulate_tampering_attack(block_index=1)
    print(f"Tamper Detected?  {tamper_result['tamper_detected']}")
    print(f"Detection Report: {tamper_result['detection_message']}")

    # 6. Post-Demonstration Validation
    print("\n--- POST-RESTORATION VERIFICATION ---")
    valid_after, msg_after = bc.is_chain_valid()
    print(f"Chain Integrity Restored? {valid_after}")

    print("\n[SUCCESS] Stage 5 Blockchain Module Verified Successfully!")
