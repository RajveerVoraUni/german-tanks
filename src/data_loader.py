"""
data_loader.py: Module for ingesting, cleaning, and extracting empirical serial numbers.
"""

import json
import os
import numpy as np
import pandas as pd


def load_d1_invoices(csv_path="data/d1_online_retail/online_retail_cleaned.csv"):
    """
    Loads cleaned Online Retail data and returns:
      dense_labels: array of 1..N_true (22,061 items)
      raw_labels: raw shifted invoice integers (with 51.2% gaps)
      thinned_labels: subset of dense labels with recorded CustomerID
      metadata: dict of summary stats
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please run initial ingestion.")

    df = pd.read_csv(csv_path, low_memory=False)
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str).str.strip()

    # Exclude cancellations and non-numeric
    valid_df = df[~df["InvoiceNo_str"].str.startswith("C")].copy()
    valid_df = valid_df[valid_df["InvoiceNo_str"].str.isdigit()].copy()
    valid_df["inv_int"] = valid_df["InvoiceNo_str"].astype(int)

    # Distinct invoice numbers sorted
    distinct_invoices = np.sort(valid_df["inv_int"].unique())
    n_true = len(distinct_invoices)
    min_inv = int(distinct_invoices[0])
    max_inv = int(distinct_invoices[-1])

    # (a) Dense labels: 1 .. N_true
    dense_labels = np.arange(1, n_true + 1)

    # (b) Raw labels: shifted so minimum is 1
    raw_labels = distinct_invoices - min_inv + 1

    # (c) Thinned customer invoices (in dense rank index)
    inv_to_rank = {inv: i + 1 for i, inv in enumerate(distinct_invoices)}
    cust_df = valid_df[valid_df["CustomerID"].notna()]
    cust_invoices = cust_df["inv_int"].unique()
    thinned_dense_labels = np.sort(np.array([inv_to_rank[inv] for inv in cust_invoices]))

    meta = {
        "n_true": n_true,
        "min_inv": min_inv,
        "max_inv": max_inv,
        "raw_max": int(raw_labels[-1]),
        "raw_span": int(raw_labels[-1]),
        "gap_count": int(raw_labels[-1] - n_true),
        "gap_fraction": float((raw_labels[-1] - n_true) / raw_labels[-1]),
        "thinned_count": len(thinned_dense_labels),
        "capture_fraction": float(len(thinned_dense_labels) / n_true),
    }

    return dense_labels, raw_labels, thinned_dense_labels, meta


def load_d5_cryptopunks(
    jsonl_path="data/d5_cryptopunks/txn_history-2021-10-07.jsonl",
):
    """
    Loads CryptoPunks transactions.
    N_true = 10,000 (tokens 0..9999 shifted to 1..10,000).
    Returns all valid token IDs and transaction dataframe.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"{jsonl_path} not found.")

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    # shift punk_id by 1 so IDs are 1..10000
    df["token_id_1indexed"] = df["punk_id"] + 1

    n_true = 10000
    dense_tokens = np.arange(1, n_true + 1)
    return dense_tokens, df
