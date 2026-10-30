"""Main Orchestration Script for Blockchain-Enabled Secure Federated Learning in IIoT.

This script executes the complete end-to-end pipeline:
1. Load and explore the Industrial IoT dataset.
2. Partition data among 3 simulated industrial clients (Factory A, Factory B, Factory C).
3. Demonstrate independent local model training.
4. Execute multi-round Federated Learning using Federated Averaging (FedAvg).
5. Record every round immutably on a lightweight SHA-256 blockchain ledger.
6. Validate blockchain integrity and chain links.
7. Conduct a controlled tampering demonstration showing real-time fraud detection.
8. Evaluate the final global model and report summary metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.blockchain import LightweightBlockchain
from clients.client import IndustrialClient
from federated_learning.fedavg import FederatedServer
from preprocessing.data_preprocessing import run_preprocessing_pipeline


def print_banner() -> None:
    """Print an informative presentation banner."""
    print("=" * 80)
    print("  BLOCKCHAIN-ENABLED SECURE FEDERATED LEARNING FOR INDUSTRIAL IoT (IIoT)")
    print("  College Mini-Project Demonstration & Evaluation Pipeline")
    print("=" * 80)


def run_pipeline(
    num_rounds: int = 5,
    local_epochs: int = 3,
    use_non_iid: bool = False,
    run_tamper_demo: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute the end-to-end collaborative intrusion detection pipeline."""
    pipeline_start_time = time.time()
    print_banner()

    # ---------------------------------------------------------
    # STAGE 1: Dataset Preprocessing and Exploration
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print("[STAGE 1] Loading and Preprocessing Industrial IoT Dataset...")
    print("#" * 80)

    dataset_path = PROJECT_ROOT / "data" / "synthetic_iiot_traffic.csv"
    prep_data = run_preprocessing_pipeline(
        dataset_path=dataset_path,
        use_non_iid=use_non_iid,
        generate_plots=generate_plots,
    )

    summary = prep_data["summary"]
    print(f"\nDataset Statistics:")
    print(f"  - Total Network Flows:      {summary['total_rows']:,}")
    print(f"  - Input Features:           {summary['total_features']}")
    print(f"  - Normal Operational Flows: {summary['normal_samples']:,} ({summary['normal_percentage']:.1f}%)")
    print(f"  - Malicious Attack Flows:   {summary['attack_samples']:,} ({summary['attack_percentage']:.1f}%)")
    print(f"  - Data Leakage Defense:     StandardScaler fitted strictly on training partition (80%).")

    # ---------------------------------------------------------
    # STAGE 2: Multi-Client Industrial Partitioning
    # ---------------------------------------------------------
    partition_mode = "Non-IID (Skewed Threat Profile)" if use_non_iid else "IID (Uniform Distribution)"
    print("\n" + "#" * 80)
    print(f"[STAGE 2] Initializing 3 Simulated Industrial Clients ({partition_mode})...")
    print("#" * 80)

    partitions = prep_data["client_partitions"]
    X_test = prep_data["X_test"]
    y_test = prep_data["y_test"]

    client_configs = [
        ("Client_1_Factory_A", "Factory A (Automotive Manufacturing)"),
        ("Client_2_Factory_B", "Factory B (Electronics Assembly)"),
        ("Client_3_Factory_C", "Factory C (Chemical / SCADA Substation)"),
    ]

    clients: List[IndustrialClient] = []
    for cid, dname in client_configs:
        client = IndustrialClient(
            client_id=cid,
            display_name=dname,
            input_dim=prep_data["num_features"],
            local_epochs=local_epochs,
        )
        pdata = partitions[cid]
        client.set_local_data(pdata["X"], pdata["y"])
        clients.append(client)
        print(
            f"  -> {dname} [{cid}]:\n"
            f"       Samples: {client.sample_count} | Normal: {client.normal_count} | "
            f"Attacks: {client.attack_count} ({client.attack_count / client.sample_count * 100:.1f}%)"
        )

    print("\n  [PRIVACY GUARANTEE] Raw network traffic remains strictly on local client devices.")

    # ---------------------------------------------------------
    # STAGE 3: Baseline Independent Local Training Demo
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print("[STAGE 3] Baseline Check: Independent Local Training (Isolated Silos)...")
    print("#" * 80)

    print("Training each factory model locally in isolation (no collaboration):")
    isolated_evaluations: Dict[str, Dict[str, Any]] = {}
    for c in clients:
        t_rep = c.train_local_model(epochs=local_epochs, verbose=False)
        e_rep = c.evaluate(X_test, y_test)
        isolated_evaluations[c.client_id] = e_rep
        print(
            f"  -> {c.display_name}:\n"
            f"       Training Time: {t_rep['training_time_sec']:.3f}s | Final Loss: {t_rep['final_loss']:.4f}\n"
            f"       Test Accuracy: {e_rep['accuracy']:.2f}% | Test F1-Score: {e_rep['f1_score']:.4f} | FPR: {e_rep['false_positive_rate']:.2f}%"
        )

    # ---------------------------------------------------------
    # STAGE 4 & 5: Federated Learning + Blockchain Integration
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"[STAGE 4 & 5] Starting Federated Learning ({num_rounds} Rounds) + Blockchain Recording...")
    print("#" * 80)

    # Initialize Blockchain Ledger
    blockchain = LightweightBlockchain()
    print(f"Blockchain Ledger Initialized:")
    print(f"  - Genesis Block Hash: {blockchain.chain[0].hash}")
    print(f"  - Previous Hash:      {blockchain.chain[0].previous_hash}")

    # Callback hook called at the end of each round to log block
    def on_round_finish(round_data: Dict[str, Any]) -> None:
        block = blockchain.record_fl_round(
            round_number=round_data["round"],
            client_hashes=round_data["client_hashes"],
            aggregated_global_hash=round_data["updated_global_hash"],
            initial_global_hash=round_data["initial_global_hash"],
            participating_clients=round_data["participating_clients"],
            client_samples=round_data["client_samples"],
            eval_metrics=round_data["eval_metrics"],
            avg_loss=round_data["avg_client_loss"],
            training_time_sec=round_data["round_duration_sec"],
        )
        print(f"  [BLOCKCHAIN] Block #{block.index} Mined & Appended:")
        print(f"    - Block Hash: {block.hash[:24]}...")
        print(f"    - Linked to Prev Hash: {block.previous_hash[:24]}...")

    # Initialize Central Federated Server
    server = FederatedServer(clients=clients, input_dim=prep_data["num_features"])

    # Run Federated Learning Pipeline
    round_history = server.run_federated_learning(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        X_test=X_test,
        y_test=y_test,
        on_round_complete=on_round_finish,
        verbose=True,
    )

    # Generate Convergence Plots
    if generate_plots:
        graphs_dir = PROJECT_ROOT / "results" / "graphs"
        server.plot_convergence_curves(graphs_dir)

    # ---------------------------------------------------------
    # STAGE 6: Blockchain Verification & Audit Check
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print("[STAGE 6] Blockchain Ledger Verification and Audit Check...")
    print("#" * 80)

    total_blocks = len(blockchain.chain)
    print(f"Total Blocks on Ledger: {total_blocks} (1 Genesis Block + {total_blocks - 1} FL Round Blocks)")

    is_valid, validation_msg = blockchain.is_chain_valid()
    print(f"Cryptographic Verification Result: {is_valid}")
    print(f"Ledger Audit Status: {validation_msg}")

    # ---------------------------------------------------------
    # STAGE 7: Controlled Tampering Attack Demonstration
    # ---------------------------------------------------------
    if run_tamper_demo and total_blocks > 1:
        print("\n" + "#" * 80)
        print("[STAGE 7] Controlled Tampering Attack Simulation (Auditing Defense)...")
        print("#" * 80)
        print("Scenario: A malicious insider attempts to modify Factory A's recorded model update in Block #1.")
        tamper_result = blockchain.simulate_tampering_attack(block_index=1)
        print(f"  -> Tamper Detected by Blockchain: {tamper_result['tamper_detected']}")
        print(f"  -> Audit Diagnostic Message:      {tamper_result['detection_message']}")
        is_clean, _ = blockchain.is_chain_valid()
        print(f"  -> Blockchain Restored to Original State: {is_clean}")

    # ---------------------------------------------------------
    # STAGE 8: Final Model Evaluation Summary
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print("[STAGE 8] Final Global Model Evaluation on Held-Out Test Set...")
    print("#" * 80)

    final_metrics = server.evaluate_global_model(X_test, y_test)
    print(f"Final Global Model Metrics:")
    print(f"  - Test Accuracy:          {final_metrics['accuracy']:.2f}%")
    print(f"  - Precision:              {final_metrics['precision']:.2f}%")
    print(f"  - Recall (Detection Rate):{final_metrics['recall']:.2f}%")
    print(f"  - F1-Score:               {final_metrics['f1_score']:.4f}")
    print(f"  - False Positive Rate:    {final_metrics['false_positive_rate']:.2f}%")
    print(f"  - Confusion Matrix:       TN={final_metrics['confusion_matrix'][0][0]}, FP={final_metrics['confusion_matrix'][0][1]}, "
          f"FN={final_metrics['confusion_matrix'][1][0]}, TP={final_metrics['confusion_matrix'][1][1]}")

    elapsed_time = time.time() - pipeline_start_time
    print("\n" + "=" * 80)
    print(f"  END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS")
    print("=" * 80)

    return {
        "prep_data": prep_data,
        "clients": clients,
        "server": server,
        "blockchain": blockchain,
        "round_history": round_history,
        "final_metrics": final_metrics,
        "isolated_evaluations": isolated_evaluations,
        "elapsed_time": elapsed_time,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Blockchain-Enabled Secure Federated Learning for IIoT pipeline."
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds (default: 5)")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per round per client (default: 3)")
    parser.add_argument("--non-iid", action="store_true", help="Use Non-IID class-skewed client partitioning")
    parser.add_argument("--no-tamper", action="store_true", help="Disable controlled tampering demonstration")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving plot figures to results/graphs/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        use_non_iid=args.non_iid,
        run_tamper_demo=not args.no_tamper,
        generate_plots=not args.no_plots,
    )
