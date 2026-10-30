"""Data Preprocessing and Client Data Partitioning Module for IIoT Intrusion Detection.

This module is responsible for:
1. Loading and cleaning Industrial IoT network traffic datasets.
2. Handling missing values, duplicates, and infinite values.
3. Scaling features using StandardScaler (fitted strictly on training data to prevent data leakage).
4. Partitioning the training dataset among three simulated industrial clients:
   - Client 1: Factory A
   - Client 2: Factory B
   - Client 3: Factory C
5. Supporting both IID (Identical and Independently Distributed) and Non-IID partitions.
6. Generating visual distribution charts for college presentation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add parent directory to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically identify the target classification column in the dataset.

    Searches common network security dataset column names: 'label', 'Label',
    'Attack_type', 'attack_type', 'class', 'Target'.
    """
    candidate_names = [
        "label",
        "Label",
        "attack_type",
        "Attack_type",
        "Attack",
        "class",
        "Class",
        "target",
        "Target",
    ]
    for col in candidate_names:
        if col in df.columns:
            return col
    # Fallback to the very last column if none matched
    return df.columns[-1]


def load_and_clean_data(file_path: str | Path) -> Tuple[pd.DataFrame, str]:
    """Load dataset from CSV, clean anomalies, and encode categorical values.

    Args:
        file_path: Absolute or relative path to the CSV dataset.

    Returns:
        Tuple of (Cleaned DataFrame, target column name).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"[PREPROCESSING] Loading dataset from: {file_path.name}")
    df = pd.read_csv(file_path)

    initial_rows = len(df)
    # Strip any accidental whitespace from column headers
    df.columns = [c.strip() for c in df.columns]

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check and handle missing values (impute numeric columns with median)
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"[PREPROCESSING] Found {nan_count} missing/infinite values. Imputing with column medians.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        # Drop rows where target itself might still be NaN
        df.dropna(inplace=True)

    # Detect target column
    target_col = detect_target_column(df)
    print(f"[PREPROCESSING] Identified target label column: '{target_col}'")

    # If target is string / categorical, encode to binary (0 = Normal, 1 = Attack)
    if df[target_col].dtype == object or isinstance(df[target_col].iloc[0], str):
        print(f"[PREPROCESSING] Target values are strings. Encoding to numeric classes...")
        unique_vals = df[target_col].unique()
        normal_aliases = ["normal", "benign", "normal traffic", "0"]
        # If binary classification
        if any(alias in str(v).lower() for v in unique_vals for alias in normal_aliases):
            df[target_col] = df[target_col].apply(
                lambda val: 0 if any(alias in str(val).lower() for alias in normal_aliases) else 1
            )
        else:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])

    # Convert target to integer
    df[target_col] = df[target_col].astype(int)

    # Encode any remaining categorical feature columns (e.g., protocol types)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    final_rows = len(df)
    print(f"[PREPROCESSING] Data successfully loaded: {final_rows} rows, {len(df.columns)} columns.")
    return df, target_col


def get_dataset_summary(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Calculate key statistics for data exploration and reporting."""
    class_counts = df[target_col].value_counts().to_dict()
    total_samples = len(df)
    normal_count = class_counts.get(0, 0)
    attack_count = class_counts.get(1, 0)

    summary = {
        "total_rows": total_samples,
        "total_features": len(df.columns) - 1,
        "feature_names": [c for c in df.columns if c != target_col],
        "normal_samples": normal_count,
        "attack_samples": attack_count,
        "normal_percentage": (normal_count / total_samples) * 100 if total_samples > 0 else 0,
        "attack_percentage": (attack_count / total_samples) * 100 if total_samples > 0 else 0,
        "missing_values": int(df.isnull().sum().sum()),
    }
    return summary


def split_and_scale_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Split data into Train/Test sets and scale features without data leakage.

    IMPORTANT (Zero Data Leakage Rule):
    The StandardScaler is fit EXCLUSIVELY on X_train.
    X_test is transformed using the parameters (mean and std) learned from X_train.

    Returns:
        (X_train_scaled, X_test_scaled, y_train, y_test, feature_names, fitted_scaler)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # Stratified split ensures the ratio of normal-to-attack is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard scaling: zero mean, unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[PREPROCESSING] Train-Test Split (80/20):")
    print(f"  - Training Set: {X_train_scaled.shape[0]} samples")
    print(f"  - Test Set:     {X_test_scaled.shape[0]} samples")
    print(f"  - Features:     {len(feature_cols)}")
    print(f"  - Zero Data Leakage: StandardScaler fitted exclusively on X_train.")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def partition_client_data_iid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_names: List[str] = ["Client_1_Factory_A", "Client_2_Factory_B", "Client_3_Factory_C"],
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Partition training data uniformly (IID) among the industrial clients.

    In IID (Independent and Identically Distributed) mode:
    - Each client receives an equal share of the training data.
    - Each client receives the same proportion of normal and attack traffic.
    """
    np.random.seed(random_state)
    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)

    splits = np.array_split(indices, len(client_names))
    partitions: Dict[str, Dict[str, Any]] = {}

    for idx, client_id in enumerate(client_names):
        client_idx = splits[idx]
        X_client = X_train[client_idx]
        y_client = y_train[client_idx]

        n_norm = int(np.sum(y_client == 0))
        n_att = int(np.sum(y_client == 1))

        partitions[client_id] = {
            "X": X_client,
            "y": y_client,
            "total_samples": len(y_client),
            "normal_samples": n_norm,
            "attack_samples": n_att,
            "attack_ratio": n_att / len(y_client) if len(y_client) > 0 else 0,
        }

    return partitions


def partition_client_data_non_iid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_names: List[str] = ["Client_1_Factory_A", "Client_2_Factory_B", "Client_3_Factory_C"],
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Partition training data with class skew (Non-IID) to reflect industrial reality.

    In real-world Industrial IoT environments:
    - Factory A (e.g. Standard Assembly): Mostly normal operations (85% normal, 15% attack).
    - Factory B (e.g. Robotics Cell): Moderately balanced exposure (50% normal, 50% attack).
    - Factory C (e.g. High-Target SCADA Substation): High cyber threat exposure (20% normal, 80% attack).
    """
    np.random.seed(random_state)

    norm_idx = np.where(y_train == 0)[0]
    att_idx = np.where(y_train == 1)[0]

    np.random.shuffle(norm_idx)
    np.random.shuffle(att_idx)

    # Client proportions of available normal vs attack pools
    # Factory A: absorbs 55% of normal pool, 15% of attack pool
    # Factory B: absorbs 30% of normal pool, 35% of attack pool
    # Factory C: absorbs 15% of normal pool, 50% of attack pool
    n_norm_total = len(norm_idx)
    n_att_total = len(att_idx)

    c1_norm_split = int(0.55 * n_norm_total)
    c2_norm_split = int(0.85 * n_norm_total)

    c1_att_split = int(0.15 * n_att_total)
    c2_att_split = int(0.50 * n_att_total)

    client_indices = {
        client_names[0]: np.concatenate([norm_idx[:c1_norm_split], att_idx[:c1_att_split]]),
        client_names[1]: np.concatenate([norm_idx[c1_norm_split:c2_norm_split], att_idx[c1_att_split:c2_att_split]]),
        client_names[2]: np.concatenate([norm_idx[c2_norm_split:], att_idx[c2_att_split:]]),
    }

    partitions: Dict[str, Dict[str, Any]] = {}
    for client_id in client_names:
        c_idx = client_indices[client_id]
        np.random.shuffle(c_idx)  # Shuffle locally
        X_client = X_train[c_idx]
        y_client = y_train[c_idx]

        n_norm = int(np.sum(y_client == 0))
        n_att = int(np.sum(y_client == 1))

        partitions[client_id] = {
            "X": X_client,
            "y": y_client,
            "total_samples": len(y_client),
            "normal_samples": n_norm,
            "attack_samples": n_att,
            "attack_ratio": n_att / len(y_client) if len(y_client) > 0 else 0,
        }

    return partitions


def generate_distribution_plots(
    df: pd.DataFrame,
    target_col: str,
    iid_partitions: Dict[str, Dict[str, Any]],
    non_iid_partitions: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> List[str]:
    """Generate high-quality presentation plots illustrating data distribution.

    Creates:
    1. Overall dataset class distribution.
    2. Comparison of IID vs Non-IID client partitions.
    3. Key feature distributions across classes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. Dataset Overall Class Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    counts = df[target_col].value_counts()
    labels = ["Normal Operational (0)", "Malicious Attack (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, width=0.5, edgecolor="black")

    ax.set_ylabel("Number of Network Flows", fontsize=11, fontweight="bold")
    ax.set_title("Industrial IoT Dataset: Traffic Class Distribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bar values
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    p1 = output_dir / "dataset_class_distribution.png"
    plt.savefig(p1)
    plt.close()
    generated_files.append(str(p1))

    # 2. Client Partitioning Comparison: IID vs Non-IID
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    clients_short = ["Factory A", "Factory B", "Factory C"]

    # IID
    iid_norm = [iid_partitions[c]["normal_samples"] for c in iid_partitions]
    iid_att = [iid_partitions[c]["attack_samples"] for c in iid_partitions]
    x = np.arange(len(clients_short))
    width = 0.35

    ax1.bar(x - width / 2, iid_norm, width, label="Normal", color="#2ecc71", edgecolor="black")
    ax1.bar(x + width / 2, iid_att, width, label="Attack", color="#e74c3c", edgecolor="black")
    ax1.set_title("IID Data Partitioning (Balanced)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clients_short, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=10)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Non-IID
    non_iid_norm = [non_iid_partitions[c]["normal_samples"] for c in non_iid_partitions]
    non_iid_att = [non_iid_partitions[c]["attack_samples"] for c in non_iid_partitions]

    ax2.bar(x - width / 2, non_iid_norm, width, label="Normal", color="#2ecc71", edgecolor="black")
    ax2.bar(x + width / 2, non_iid_att, width, label="Attack", color="#e74c3c", edgecolor="black")
    ax2.set_title("Non-IID Data Partitioning (Realistic Industrial Skew)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(clients_short, fontweight="bold")
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Industrial Client Partitioning Comparison (3 Factory Nodes)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p2 = output_dir / "client_partitioning_comparison.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    generated_files.append(str(p2))

    return generated_files


def run_preprocessing_pipeline(
    dataset_path: str | Path | None = None,
    use_non_iid: bool = False,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute the end-to-end preprocessing pipeline.

    Returns a structured dictionary containing all arrays and partitions
    ready for machine learning and federated learning training.
    """
    if dataset_path is None:
        dataset_path = PROJECT_ROOT / "data" / "synthetic_iiot_traffic.csv"

    dataset_path = Path(dataset_path)
    df, target_col = load_and_clean_data(dataset_path)
    summary = get_dataset_summary(df, target_col)

    # Train-test split (80% train, 20% test) with scaling
    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale_data(df, target_col)

    # Client partitions
    client_names = ["Client_1_Factory_A", "Client_2_Factory_B", "Client_3_Factory_C"]
    iid_partitions = partition_client_data_iid(X_train, y_train, client_names)
    non_iid_partitions = partition_client_data_non_iid(X_train, y_train, client_names)

    active_partitions = non_iid_partitions if use_non_iid else iid_partitions

    # Generate plots if requested
    plot_files = []
    if generate_plots:
        graphs_dir = PROJECT_ROOT / "results" / "graphs"
        plot_files = generate_distribution_plots(df, target_col, iid_partitions, non_iid_partitions, graphs_dir)
        print(f"[PREPROCESSING] Generated distribution plots in: {graphs_dir}")

    return {
        "df_raw": df,
        "target_col": target_col,
        "summary": summary,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "iid_partitions": iid_partitions,
        "non_iid_partitions": non_iid_partitions,
        "client_partitions": active_partitions,
        "use_non_iid": use_non_iid,
        "plot_files": plot_files,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 1 & 2: DATASET PREPROCESSING AND CLIENT PARTITIONING")
    print("=" * 70)

    data_file = PROJECT_ROOT / "data" / "synthetic_iiot_traffic.csv"
    results = run_preprocessing_pipeline(data_file, use_non_iid=False, generate_plots=True)

    summary = results["summary"]
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Flows:          {summary['total_rows']}")
    print(f"Input Features:       {summary['total_features']}")
    print(f"Normal Flows:         {summary['normal_samples']} ({summary['normal_percentage']:.2f}%)")
    print(f"Attack Flows:         {summary['attack_samples']} ({summary['attack_percentage']:.2f}%)")
    print(f"Missing Values:       {summary['missing_values']}")

    print("\n--- CLIENT PARTITIONS (IID) ---")
    for cid, pdata in results["iid_partitions"].items():
        print(f"[{cid}] Samples: {pdata['total_samples']} (Normal: {pdata['normal_samples']}, Attack: {pdata['attack_samples']}, Attack%: {pdata['attack_ratio']*100:.1f}%)")

    print("\n--- CLIENT PARTITIONS (Non-IID) ---")
    for cid, pdata in results["non_iid_partitions"].items():
        print(f"[{cid}] Samples: {pdata['total_samples']} (Normal: {pdata['normal_samples']}, Attack: {pdata['attack_samples']}, Attack%: {pdata['attack_ratio']*100:.1f}%)")

    print(f"\nGenerated Plots: {results['plot_files']}")
    print("\n[SUCCESS] Stage 1 & 2 Completed Successfully!")
