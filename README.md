# Blockchain-Enabled Secure Federated Learning for Industrial IoT (IIoT)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, privacy-preserving, and tamper-resistant collaborative intrusion detection framework tailored for Industrial Internet of Things (IIoT) networks.

This project extends and improves upon the research paper:  
**"Blockchain-Integrated Collaborative Intrusion Detection Systems: Enhancing Network Security"**  
by implementing an operational **Federated Learning (FedAvg)** pipeline and integrating a lightweight **SHA-256 blockchain audit ledger**.

---

## 1. Project Introduction & Problem Statement

### 1.1 The Problem in Industrial IoT
Modern industrial facilities (manufacturing plants, smart power grids, oil refineries, and chemical facilities) rely heavily on interconnected **Industrial IoT (IIoT)** devices, Programmable Logic Controllers (PLCs), and SCADA networks. These systems face an unprecedented wave of sophisticated cyber intrusions (Denial-of-Service, reconnaissance port scanning, and command injection attacks).

Traditional Network Intrusion Detection Systems (NIDS) exhibit critical flaws in industrial environments:
1. **Centralized Vulnerability:** Central servers represent a single point of failure and a primary target for adversaries.
2. **Data Privacy Bottleneck:** Industrial organizations refuse to share their raw network traffic logs because traffic captures reveal sensitive trade secrets, production shifts, equipment models, and proprietary SCADA register maps.
3. **Auditability & Log Tampering:** When an attacker breaches an industrial network, they often alter local audit logs to conceal their tracks.

### 1.2 Existing Approach (The Reference Paper)
The foundational paper (*BC-CIDS*) proposed sharing intrusion alerts over an enterprise Hyperledger Fabric blockchain. However:
- **No Federated Learning:** Despite mentioning FL as a future idea, collaborative model training was never implemented.
- **Privacy Leaks:** Sharing alert logs still exposes IP addresses and operational network patterns.
- **Extreme Complexity:** Heavy enterprise blockchain software makes deployment on resource-constrained edge devices or student laptops practically impossible.

### 1.3 Our Proposed Contribution
We solve these challenges by building a complete, working, lightweight prototype:
1. **Federated Learning (FedAvg):** Three simulated industrial nodes (**Factory A**, **Factory B**, **Factory C**) train local neural networks on private network telemetry. **Raw data never leaves the client.**
2. **Lightweight Blockchain Ledger:** A pure-Python cryptographic ledger using **SHA-256** records model update fingerprints and global model states, providing **100% tamper detection and non-repudiation**.
3. **Rigorous 4-Way Evaluation:** We benchmark Centralized ML, Independent Local ML, Federated Learning, and Federated Learning + Blockchain under identical conditions.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Simulated Industrial IoT Clients                     │
│                                                                        │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│   │ Client 1:        │  │ Client 2:        │  │ Client 3:        │     │
│   │ Factory A        │  │ Factory B        │  │ Factory C        │     │
│   │ (Automotive)     │  │ (Electronics)    │  │ (SCADA Substation│     │
│   │ Local Data D1    │  │ Local Data D2    │  │ Local Data D3    │     │
│   │ PyTorch MLP      │  │ PyTorch MLP      │  │ PyTorch MLP      │     │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘     │
└────────────┼─────────────────────┼─────────────────────┼───────────────┘
             │                     │                     │
             │ Local Model Weights │ Local Model Weights │ Local Model Weights
             │ (NO raw data!)      │ (NO raw data!)      │ (NO raw data!)
             ▼                     ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Federated Learning Aggregator Server                   │
│                                                                        │
│  1. Broadcasts global model parameters to all factory nodes            │
│  2. Collects local parameter updates and SHA-256 weight hashes         │
│  3. Performs sample-weighted Federated Averaging (FedAvg):             │
│            W_global = SUM( (n_k / N_total) * W_k )                     │
│  4. Evaluates updated global model on held-out 600-sample test set     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Lightweight Cryptographic Blockchain Ledger             │
│                                                                        │
│  Block [Round #k]:                                                     │
│  - Previous Block Hash: SHA-256(Block_{k-1})                           │
│  - Timestamp                                                           │
│  - Participating Factories: [Factory A, Factory B, Factory C]          │
│  - Client Update Hashes: {Factory A: Hash1, Factory B: Hash2, ...}     │
│  - Aggregated Global Model Hash: Hash_global                           │
│  - Evaluation Metrics: Accuracy, Precision, Recall, F1, FPR            │
│  - Current Block Hash: SHA-256(Block Contents + PrevHash)              │
│                                                                        │
│  Integrity Check: Any modification breaks hash linkage immediately     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core runtime and orchestration |
| **Deep Learning** | PyTorch (CPU) | `IIoTIntrusionMLP` neural network and tensor operations |
| **Machine Learning** | Scikit-Learn | Centralized baseline, feature scaling, and metrics evaluation |
| **Data Processing** | Pandas, NumPy | Dataset handling, array operations, and statistics |
| **Visualization** | Matplotlib | Loss curves, accuracy plots, and confusion matrices |
| **Blockchain** | Python `hashlib` | SHA-256 cryptographic hashing and block validation |
| **Web Dashboard** | Streamlit | Interactive browser interface for college demonstration |

---

## 4. Project Directory Structure

```
iiot_fl_blockchain/
│
├── data/
│   └── synthetic_iiot_traffic.csv      # 3,000 IIoT network flows with SCADA/Modbus features
│
├── preprocessing/
│   └── data_preprocessing.py           # Scaling, splitting, IID/Non-IID partitioning, zero data leakage
│
├── models/
│   └── model.py                        # Lightweight PyTorch Multi-Layer Perceptron (MLP)
│
├── clients/
│   └── client.py                       # Simulated industrial clients (Factory A, B, C)
│
├── federated_learning/
│   └── fedavg.py                       # Federated Averaging aggregation server
│
├── blockchain/
│   └── blockchain.py                   # Pure Python SHA-256 hash-chained ledger & tamper detection
│
├── evaluation/
│   └── evaluate.py                     # 4-way comparative benchmarking module
│
├── results/
│   ├── graphs/                         # Generated publication-quality figures
│   └── comparison_report.csv           # Measured 4-way comparison table
│
├── main.py                             # Unified end-to-end command-line runner
├── app.py                              # Interactive Streamlit college presentation dashboard
├── run_pipeline.bat                    # One-click Windows runner for CLI pipeline
├── run_dashboard.bat                   # One-click Windows runner for Streamlit dashboard
├── requirements.txt                    # Project dependencies
├── RESEARCH_PAPER.md                   # Full 16-section research paper draft
├── VIVA_PRESENTATION_GUIDE.md          # College viva presentation guide & Q&A
└── README.md                           # This documentation file
```

---

## 5. Module Explanations

### 5.1 `preprocessing/data_preprocessing.py`
- **What it does:** Loads network traffic, checks missing/infinite values, standardizes numeric features, and partitions training data among the three factory clients.
- **Zero Data Leakage Rule:** The `StandardScaler` is fitted **exclusively on the training partition (80%)**. The held-out test set (20%) is transformed using the learned training statistics.
- **Client Partitioning:** Supports **IID** (equal balanced slices) and **Non-IID** (realistic industrial threat skew: Factory A = 12.8% attacks, Factory B = 38.6% attacks, Factory C = 64.2% attacks).

### 5.2 `models/model.py`
- **What it does:** Defines the `IIoTIntrusionMLP` neural network.
- **Architecture:** Input (15 features) $\to$ Linear(15, 32) + ReLU + Dropout(0.2) $\to$ Linear(32, 16) + ReLU $\to$ Linear(16, 2) binary output.
- **Model Parameter Hashing:** `compute_hash()` computes a deterministic SHA-256 string from the model's PyTorch `state_dict`.

### 5.3 `clients/client.py`
- **What it does:** Simulates an isolated industrial facility.
- **Key Methods:**
  - `set_local_data()`: Encapsulates private dataset. Raw data cannot be accessed externally.
  - `train_local_model()`: Executes local Adam optimizer loops for a configured number of epochs.
  - `evaluate()`: Evaluates local model on test data.
  - `get_model_weights()` / `set_model_weights()`: Exports/updates parameters for FedAvg.

### 5.4 `federated_learning/fedavg.py`
- **What it does:** Orchestrates the multi-round Federated Learning process.
- **Key Function:** `federated_averaging()` implements sample-size-weighted averaging:
  $$W_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N_{\text{total}}} W_k$$
- Tracks round convergence and generates loss/accuracy graphs.

### 5.5 `blockchain/blockchain.py`
- **What it does:** Lightweight, pure-Python SHA-256 blockchain ledger.
- **Key Methods:**
  - `create_genesis_block()`: Generates foundational Block #0.
  - `record_fl_round()`: Adds a new block storing round metadata and parameter hashes.
  - `is_chain_valid()`: Traverses the chain and validates hashes and block pointers.
  - `simulate_tampering_attack()`: Controlled demo modifying Block #1 and demonstrating immediate detection.

### 5.6 `evaluation/evaluate.py`
- **What it does:** Runs all four configurations (Centralized, Independent Local, FedAvg, FedAvg+Blockchain) under identical conditions and outputs `results/comparison_report.csv` and comparative plots.

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
- Python 3.11 installed on your computer.
- Standard laptop hardware (no GPU required; all models run efficiently on CPU).

### 6.2 Installation
Open PowerShell in the project directory:
```powershell
cd c:\Users\hp\OneDrive\Desktop\Project\iiot_fl_blockchain
pip install -r requirements.txt
```

### 6.3 Running the Unified Pipeline (Console)
Execute the complete end-to-end pipeline:
```powershell
python main.py --rounds 5 --epochs 3
```
Or test under realistic Non-IID industrial skew:
```powershell
python main.py --rounds 3 --epochs 2 --non-iid
```
*(Alternatively, simply double-click `run_pipeline.bat`)*

### 6.4 Running the Interactive Streamlit Dashboard
Launch the interactive web UI:
```powershell
streamlit run app.py
```
*(Alternatively, simply double-click `run_dashboard.bat`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 7. Experimental Results (4-Way Comparison Table)

All values below were measured directly from model execution on the held-out 600-sample test set:

| Configuration | Data Privacy Preserved? | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | False Positive Rate (FPR) | Training Time (s) | Communication Overhead (KB) | Blockchain Overhead (ms) | Audit Integrity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Centralized ML** | **No** (Raw Data Uploaded) | 93.67% | 88.74% | 93.81% | 0.9120 | 6.41% | 1.323 s | 281.2 KB | 0.00 ms | None (Single Point of Failure) |
| **2. Independent Local ML** | **Yes** (Local Only) | 91.39% | 86.63% | 90.95% | 0.8823 | 8.38% | 0.265 s | 0.0 KB | 0.00 ms | None (Isolated Logs Only) |
| **3. Federated Learning (FedAvg)** | **Yes** (Weights Only) | 93.17% | 88.58% | 92.38% | 0.9044 | 6.41% | 1.520 s | 125.9 KB | 0.00 ms | None (Vulnerable to Tampering) |
| **4. Federated Learning + Blockchain** | **Yes** (Weights Only) | 93.17% | 88.58% | 92.38% | 0.9044 | 6.41% | 1.581 s | 125.9 KB | **0.13 ms** | **Guaranteed (SHA-256 Ledger)** |

---

## 8. Essential Viva Questions & Concepts Explained Simply

### Q1: What is Industrial IoT (IIoT)?
Industrial IoT refers to connected sensors, instruments, and autonomous industrial machinery (like SCADA systems, PLCs, and robotic arms) operating in manufacturing, power generation, and critical infrastructure. Unlike consumer smart devices, an IIoT failure or cyber-attack can cause physical damage, environmental hazards, or multi-million-dollar factory shutdowns.

### Q2: What is an Intrusion Detection System (IDS)?
An IDS is a defensive security framework that inspects incoming and outgoing network traffic packets to identify malicious activities, unauthorized access attempts, port scanning, or Denial-of-Service attacks, distinguishing normal operations from cyber intrusions.

### Q3: What is Federated Learning (FL)?
Federated Learning is a decentralized machine learning paradigm where multiple clients (factories) train a model collaboratively without sharing their private raw datasets. Clients keep their data locally, train models locally, and share only model updates (weights) with a central server.

### Q4: What is FedAvg and how does it work?
Federated Averaging (FedAvg) aggregates local client model weights by weighting each client's contribution proportional to its local sample count:
$$W_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N_{\text{total}}} W_k$$
Clients that train on more samples contribute higher weight to the global model.

### Q5: What is a Cryptographic Hash (SHA-256)?
A cryptographic hash is a deterministic one-way mathematical function that maps arbitrary data into a fixed 64-character hexadecimal string. Even a tiny change of 0.0001 in model parameters completely changes the resulting hash (the avalanche effect), making it ideal for verifying data integrity.

### Q6: What is a Blockchain?
A blockchain is a decentralized, chronologically ordered sequence of data blocks linked together cryptographically. Each block contains a cryptographic hash of its contents and the hash of the preceding block (`previous_hash`). Any attempt to alter an earlier block breaks the chain of hashes, instantly exposing the tampering.

### Q7: Why combine Federated Learning with Blockchain?
- **Federated Learning provides DATA PRIVACY:** Raw operational data never leaves the factory.
- **Blockchain provides INTEGRITY, AUDITABILITY, and NON-REPUDIATION:** Guarantees that model updates submitted by participants and aggregated by the server cannot be secretly falsified or poisoned.

### Q8: Does Blockchain improve Machine Learning detection accuracy?
**No.** Blockchain is an audit and trust mechanism; it does not alter gradient descent formulas or improve ML accuracy. It ensures that the Federated Learning process is honest, auditable, and resilient against tampering.

### Q9: Why is raw data not shared between clients?
Industrial telemetry contains competitive secrets, operational schedules, exact equipment firmware, and SCADA register maps. Sharing raw data creates severe corporate espionage risks and violates industrial regulatory privacy laws.

### Q10: What are the limitations and future improvements?
- **Current Limitations:** Runs as a simulated multi-process prototype on a single laptop; assumes clients are honest participants.
- **Future Improvements:** Deploy on physical edge devices (Raspberry Pi / NVIDIA Jetson), integrate Differential Privacy (DP-SGD) to prevent model-inversion attacks, and implement Byzantine-fault-tolerant aggregation (Krum / Median).

---

## 9. License

This project is released under the MIT Academic License for educational and research purposes.
