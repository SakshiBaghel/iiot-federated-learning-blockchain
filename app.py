"""Interactive Streamlit Demonstration Dashboard for IIoT Secure Federated Learning & Blockchain.

Designed for college project presentations, viva demonstrations, and professor reviews:
1. Dataset Explorer and Multi-Client Industrial Partitioning.
2. Federated Learning Simulation using FedAvg.
3. Cryptographic Blockchain Explorer and Interactive Tamper Lab.
4. Comprehensive 4-Way Evaluation Benchmarks and Plots.
5. Complete Viva Presentation Q&A Guide.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blockchain.blockchain import LightweightBlockchain
from clients.client import IndustrialClient
from evaluation.evaluate import run_comparative_experiments
from federated_learning.fedavg import FederatedServer
from preprocessing.data_preprocessing import run_preprocessing_pipeline

# Configure Streamlit Page
st.set_page_config(
    page_title="IIoT Federated Learning & Blockchain",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Theme-Adaptive & Dark Mode Compatible)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        margin-bottom: 12px;
        color: #e2e8f0;
    }
    .metric-card h4 {
        color: #38bdf8 !important;
        margin-top: 0;
        margin-bottom: 6px;
    }
    .metric-card p {
        color: #94a3b8 !important;
        margin-bottom: 8px;
    }
    .metric-card b {
        color: #f8fafc !important;
    }
    .status-verified {
        background-color: #0f291e;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 16px;
        color: #4ade80;
    }
    .status-tampered {
        background-color: #2a1517;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 16px;
        color: #f87171;
    }
    .tamper-box {
        background-color: #2a1517;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        color: #fca5a5;
    }
    .tamper-box b {
        color: #f87171 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_cached_preprocessing(use_non_iid: bool):
    """Cache data preprocessing to ensure snappy dashboard interactions."""
    return run_preprocessing_pipeline(generate_plots=True, use_non_iid=use_non_iid)


@st.cache_data
def load_comparison_report():
    """Load comparison CSV if available or generate it."""
    csv_path = PROJECT_ROOT / "results" / "comparison_report.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return run_comparative_experiments(use_non_iid=True, num_rounds=5, local_epochs=3)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/industrial-robot.png", width=64)
    st.title("Project Controls")
    st.markdown("**IIoT Intrusion Detection System**")
    st.markdown("---")

    rounds_slider = st.slider("FL Communication Rounds", min_value=1, max_value=10, value=5)
    epochs_slider = st.slider("Local Epochs per Client", min_value=1, max_value=5, value=3)
    partition_mode = st.radio("Client Data Distribution", ["IID (Balanced)", "Non-IID (Realistic Skew)"])
    use_non_iid = partition_mode.startswith("Non-IID")

    st.markdown("---")
    st.markdown("### Technology Stack")
    st.caption("• **Python 3.11** Core Engine")
    st.caption("• **PyTorch** Multi-Layer Perceptron")
    st.caption("• **FedAvg** Weighted Aggregation")
    st.caption("• **SHA-256** Simulated Blockchain")
    st.caption("• **Scikit-Learn** Metrics & Baseline")
    st.caption("• **Streamlit** Interactive UI")
    st.markdown("---")
    st.caption("College Mini-Project Academic Prototype")

# Main Header
st.markdown('<div class="main-header">🛡️ Blockchain-Enabled Secure Federated Learning for IIoT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Collaborative, Privacy-Preserving Intrusion Detection with Cryptographic Auditability</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset & Clients",
    "🤝 Federated Learning (FedAvg)",
    "⛓️ Blockchain & Tamper Lab",
    "📈 4-Way Comparison Report",
    "🎓 Viva & Presentation Guide",
])

# ------------------------------------------------------------------------------------------------
# TAB 1: Dataset & Industrial Clients
# ------------------------------------------------------------------------------------------------
with tab1:
    st.header("1. Industrial IoT Dataset & Factory Clients")
    st.markdown(
        "Each participating factory monitors its local Industrial IoT (IIoT) sensors, PLCs, and SCADA "
        "network traffic. **Raw traffic never leaves the local factory premises.**"
    )

    data_bundle = load_cached_preprocessing(use_non_iid)
    summary = data_bundle["summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Network Flows", f"{summary['total_rows']:,}")
    col2.metric("Input Features", summary["total_features"])
    col3.metric("Normal Traffic Flows", f"{summary['normal_samples']:,} ({summary['normal_percentage']:.1f}%)")
    col4.metric("Attack Traffic Flows", f"{summary['attack_samples']:,} ({summary['attack_percentage']:.1f}%)")

    st.markdown("---")
    st.subheader("Simulated Industrial Entities")
    client_cols = st.columns(3)

    partitions = data_bundle["client_partitions"]
    configs = [
        ("Client_1_Factory_A", "🏭 Factory A (Automotive)", "Standard automotive manufacturing line with robot arms."),
        ("Client_2_Factory_B", "🔬 Factory B (Electronics)", "Electronics testing and cleanroom assembly operations."),
        ("Client_3_Factory_C", "⚡ Factory C (SCADA Substation)", "High-risk chemical and power grid SCADA control substation."),
    ]

    for idx, (cid, name, desc) in enumerate(configs):
        pdata = partitions[cid]
        with client_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <p style="font-size:0.88rem; color:#94a3b8; margin-bottom:8px;">{desc}</p>
                    <hr style="border:none; border-top:1px solid #334155; margin:8px 0;"/>
                    <div style="font-size:0.92rem; line-height:1.6; color:#e2e8f0;">
                        <b style="color:#f8fafc;">Local Samples:</b> {pdata['total_samples']}<br/>
                        <b style="color:#f8fafc;">Normal Traffic:</b> {pdata['normal_samples']} ({ (pdata['normal_samples']/pdata['total_samples'])*100:.1f}%)<br/>
                        <b style="color:#f8fafc;">Attack Traffic:</b> {pdata['attack_samples']} ({ (pdata['attack_samples']/pdata['total_samples'])*100:.1f}%)
                    </div>
                    <div style="margin-top:10px; color:#4ade80; font-weight:bold; font-size:0.85rem;">
                        🔒 Raw Data Isolated Locally
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Dataset Visualizations")
    img_col1, img_col2 = st.columns(2)
    p1 = PROJECT_ROOT / "results" / "graphs" / "dataset_class_distribution.png"
    p2 = PROJECT_ROOT / "results" / "graphs" / "client_partitioning_comparison.png"

    if p1.exists():
        img_col1.image(str(p1), caption="Figure 1: Target Class Balance (Normal vs. Attack)", use_container_width=True)
    if p2.exists():
        img_col2.image(str(p2), caption="Figure 2: IID vs Non-IID Distribution Across Clients", use_container_width=True)


# ------------------------------------------------------------------------------------------------
# TAB 2: Federated Learning Simulation (FedAvg)
# ------------------------------------------------------------------------------------------------
with tab2:
    st.header("2. Federated Learning with Federated Averaging (FedAvg)")
    st.markdown(
        "Instead of sending private network captures to the cloud, each factory trains a local PyTorch neural network "
        "and sends **only its model parameters (weights)**. The central aggregator combines them using sample-weighted FedAvg:"
    )

    st.latex(r"\mathbf{W}_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N_{\text{total}}} \mathbf{W}_k")

    st.markdown("---")
    if st.button("🚀 Run Live Federated Learning Simulation", type="primary"):
        with st.spinner("Simulating multi-client Federated Learning rounds..."):
            # Setup clients
            client_configs = [
                ("Client_1_Factory_A", "Factory A"),
                ("Client_2_Factory_B", "Factory B"),
                ("Client_3_Factory_C", "Factory C"),
            ]
            live_clients = []
            for cid, dname in client_configs:
                cl = IndustrialClient(cid, dname, input_dim=data_bundle["num_features"], local_epochs=epochs_slider)
                cl.set_local_data(partitions[cid]["X"], partitions[cid]["y"])
                live_clients.append(cl)

            live_server = FederatedServer(live_clients, input_dim=data_bundle["num_features"])
            live_blockchain = LightweightBlockchain()

            def live_bc_hook(rdata):
                live_blockchain.record_fl_round(
                    round_number=rdata["round"],
                    client_hashes=rdata["client_hashes"],
                    aggregated_global_hash=rdata["updated_global_hash"],
                    initial_global_hash=rdata["initial_global_hash"],
                    participating_clients=rdata["participating_clients"],
                    client_samples=rdata["client_samples"],
                    eval_metrics=rdata["eval_metrics"],
                    avg_loss=rdata["avg_client_loss"],
                    training_time_sec=rdata["round_duration_sec"],
                )

            history = live_server.run_federated_learning(
                num_rounds=rounds_slider,
                local_epochs=epochs_slider,
                X_test=data_bundle["X_test"],
                y_test=data_bundle["y_test"],
                on_round_complete=live_bc_hook,
                verbose=False,
            )
            st.session_state["live_history"] = history
            st.session_state["live_blockchain"] = live_blockchain

        st.success(f"Successfully simulated {rounds_slider} communication rounds!")

    # Display Round History
    if "live_history" in st.session_state:
        history = st.session_state["live_history"]
        st.subheader("Communication Rounds Summary")
        table_rows = []
        for r in history:
            metrics = r["eval_metrics"]
            table_rows.append({
                "Round": f"Round #{r['round']}",
                "Avg Client Loss": f"{r['avg_client_loss']:.4f}",
                "Test Accuracy": f"{metrics.get('accuracy', 0):.2f}%",
                "Precision": f"{metrics.get('precision', 0):.2f}%",
                "Recall (Detection)": f"{metrics.get('recall', 0):.2f}%",
                "F1-Score": f"{metrics.get('f1_score', 0):.4f}",
                "False Positive Rate": f"{metrics.get('false_positive_rate', 0):.2f}%",
                "Global Model Hash (SHA-256)": r["updated_global_hash"][:18] + "...",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.subheader("Convergence Dynamics")
        c_plot = PROJECT_ROOT / "results" / "graphs" / "fedavg_convergence_curves.png"
        if c_plot.exists():
            st.image(str(c_plot), caption="Figure 3: Global Test Accuracy and Training Loss Across Communication Rounds", use_container_width=True)


# ------------------------------------------------------------------------------------------------
# TAB 3: Blockchain Explorer & Tamper Lab
# ------------------------------------------------------------------------------------------------
with tab3:
    st.header("3. Lightweight Cryptographic Blockchain & Tamper Lab")
    st.markdown(
        "For each communication round, the aggregator writes a new block containing the SHA-256 digital fingerprint "
        "of each factory's model parameters and the aggregated global model. **No raw traffic is stored on the ledger.**"
    )

    bc: LightweightBlockchain = st.session_state.get("live_blockchain", None)
    if bc is None:
        bc = LightweightBlockchain()
        # Seed 2 demonstration rounds
        h_demo = {"Client_1_Factory_A": "40861ea0678b02b1" + "0" * 48, "Client_2_Factory_B": "763a0d7fe78f3222" + "0" * 48}
        bc.record_fl_round(1, h_demo, "0ac6b1f383f5427e" + "0" * 48, "a6975553ca1f8550" + "0" * 48,
                           ["Client_1_Factory_A", "Client_2_Factory_B"], {"Client_1_Factory_A": 800, "Client_2_Factory_B": 800},
                           {"accuracy": 100.0, "f1_score": 1.0}, 0.015, 0.45)
        bc.record_fl_round(2, h_demo, "5e6b00122c0ee58c" + "0" * 48, "0ac6b1f383f5427e" + "0" * 48,
                           ["Client_1_Factory_A", "Client_2_Factory_B"], {"Client_1_Factory_A": 800, "Client_2_Factory_B": 800},
                           {"accuracy": 100.0, "f1_score": 1.0}, 0.005, 0.42)

    is_valid, validation_msg = bc.is_chain_valid()
    if is_valid:
        st.markdown(
            f"""
            <div class="status-verified">
                <b style="font-size:1.05rem;">Blockchain Ledger Status: ✅ VERIFIED & IMMUTABLE</b><br/>
                <span style="color:#cbd5e1; font-size:0.88rem;">{validation_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-tampered">
                <b style="font-size:1.05rem;">Blockchain Ledger Status: 🚨 TAMPERING DETECTED</b><br/>
                <span style="color:#fca5a5; font-size:0.88rem;">{validation_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader(f"Blocks on Chain ({len(bc.chain)} Total)")
    for b in bc.chain:
        with st.expander(f"📦 Block #{b.index} | Hash: {b.hash[:16]}... | PrevHash: {b.previous_hash[:16]}...", expanded=(b.index == 0)):
            st.json(b.to_dict())

    st.markdown("---")
    st.subheader("🧪 Interactive Tampering Attack Demonstration")
    st.markdown(
        "Demonstrate why blockchain is tamper-evident: if a rogue insider attempts to alter an old block's "
        "recorded model update hash, the block's recalculated hash will immediately mismatch, breaking the chain."
    )

    tamper_col1, tamper_col2 = st.columns([2, 1])
    with tamper_col1:
        block_idx_to_tamper = st.number_input("Select Block to Tamper With", min_value=1, max_value=max(1, len(bc.chain) - 1), value=1)
    with tamper_col2:
        st.write("")
        st.write("")
        tamper_btn = st.button("⚠️ Attempt Malicious Alteration", type="secondary")

    if tamper_btn:
        res = bc.simulate_tampering_attack(block_index=int(block_idx_to_tamper))
        st.error(f"🚨 {res['detection_message']}")
        st.markdown(
            f"""
            <div class="tamper-box">
                <b style="font-size:1.05rem; color:#f87171;">🛡️ Cryptographic Audit Defense Succeeded:</b><br/>
                <div style="color:#fca5a5; font-size:0.95rem; margin-top:6px; margin-bottom:4px;">
                    The ledger detected that <b>Block #{block_idx_to_tamper}</b>'s content was modified.
                </div>
                <div style="color:#e2e8f0; font-size:0.9rem;">
                    Because SHA-256 hash functions are one-way and avalanche-sensitive, unauthorized alterations break the chain immediately!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------------------------
# TAB 4: 4-Way Comparative Evaluation
# ------------------------------------------------------------------------------------------------
with tab4:
    st.header("4. Four-Way Comparative Performance Evaluation")
    st.markdown(
        "We quantitatively evaluate the four core configurations on the exact same test dataset. "
        "All values below are measured directly from model execution:"
    )

    df_comp = load_comparison_report()
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Evaluation Visualizations")

    g1, g2 = st.columns(2)
    p_comp1 = PROJECT_ROOT / "results" / "graphs" / "four_way_metrics_comparison.png"
    p_comp2 = PROJECT_ROOT / "results" / "graphs" / "confusion_matrices_comparison.png"
    p_comp3 = PROJECT_ROOT / "results" / "graphs" / "overhead_and_security_comparison.png"

    if p_comp1.exists():
        g1.image(str(p_comp1), caption="Figure 4: Metric Comparison Across 4 System Approaches", use_container_width=True)
    if p_comp2.exists():
        g2.image(str(p_comp2), caption="Figure 5: Confusion Matrices on 600-Sample Test Set", use_container_width=True)

    if p_comp3.exists():
        st.image(str(p_comp3), caption="Figure 6: Computational Overhead and False Positive Rate Trade-Offs", use_container_width=True)


# ------------------------------------------------------------------------------------------------
# TAB 5: College Viva & Presentation Q&A Guide
# ------------------------------------------------------------------------------------------------
with tab5:
    st.header("5. College Viva & Presentation Q&A Guide")
    st.markdown("Use these concise, professor-ready answers during your project review or viva:")

    qas = [
        ("1. What is Industrial IoT (IIoT)?",
         "Industrial IoT refers to connected sensors, instruments, and autonomous industrial machinery (like SCADA systems, PLCs, and robotic arms) operating in manufacturing, power generation, and critical infrastructure. Unlike consumer smart devices, an IIoT failure or cyber-attack can cause physical damage, environmental hazards, or multi-million-dollar factory shutdowns."),

        ("2. What is an Intrusion Detection System (IDS)?",
         "An IDS is a defensive security framework that inspects incoming and outgoing network traffic packets to identify malicious activities, unauthorized access attempts, port scanning, or Denial-of-Service attacks, distinguishing normal operations from cyber intrusions."),

        ("3. What is Federated Learning (FL)?",
         "Federated Learning is a decentralized machine learning paradigm where multiple clients (factories) train a model collaboratively without sharing their private raw datasets. Clients keep their data locally, train models locally, and share only model updates (weights) with a central server."),

        ("4. What is FedAvg and how does it work?",
         "Federated Averaging (FedAvg) aggregates local client model weights by weighting each client's contribution proportional to its local sample count: W_global = SUM((n_k / N_total) * W_k). Clients that train on more samples contribute higher weight to the global model."),

        ("5. What is a Cryptographic Hash (SHA-256)?",
         "A cryptographic hash is a deterministic one-way mathematical function that maps arbitrary data into a fixed 64-character hexadecimal string. Even a tiny change of 0.0001 in model parameters completely changes the resulting hash (the avalanche effect), making it ideal for verifying data integrity."),

        ("6. What is a Blockchain?",
         "A blockchain is a decentralized, chronologically ordered sequence of data blocks linked together cryptographically. Each block contains a cryptographic hash of its contents and the hash of the preceding block ('previous_hash'). Any attempt to alter an earlier block breaks the chain of hashes, instantly exposing the tampering."),

        ("7. Why combine Federated Learning with Blockchain?",
         "Federated Learning provides DATA PRIVACY (raw operational data never leaves the factory). Blockchain provides INTEGRITY, AUDITABILITY, and NON-REPUDIATION (guarantees that model updates submitted by participants and aggregated by the server cannot be secretly falsified or poisoned)."),

        ("8. Does Blockchain improve Machine Learning detection accuracy?",
         "No, and this is an important scientific distinction. Blockchain is an audit and trust mechanism; it does not alter gradient descent formulas or improve ML accuracy. It ensures that the Federated Learning process is honest, auditable, and resilient against tampering."),

        ("9. What are the limitations of our prototype and future scope?",
         "Our prototype simulates network communication and blockchain ledger state on a single machine using Python and PyTorch. In future work, this can be deployed across physically distributed Raspberry Pi edge nodes, incorporating Differential Privacy noise injection, and smart contracts for automated IP-blocking."),
    ]

    for q, a in qas:
        with st.expander(f"❓ {q}", expanded=False):
            st.markdown(a)
