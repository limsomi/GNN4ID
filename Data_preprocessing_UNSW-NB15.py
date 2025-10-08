import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_and_split(input_csv='data/NF-UNSW-NB15-v4.csv',
                         out_dir='data/processed_unsw',
                         test_size=0.2, val_size=0.2,
                         random_state=42, balance=False, balance_per_class=20000):
    """
    Preprocesses the input CSV file and splits it into train, validation, and test datasets.

    Args:
        input_csv (str): Path to the input CSV file.
        out_dir (str): Directory to save the processed datasets.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
        balance (bool): Whether to balance the dataset by class.
        balance_per_class (int): Number of samples per class for balancing.

    Returns:
        None: Saves the train, validation, and test datasets as CSV files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "shape:", df.shape)

    # Normalize column names to uppercase for robust mapping
    df.columns = [c.strip() for c in df.columns]
    colset = set(df.columns)

    # Ensure Label column exists (prefer 'Label', else map 'Attack')
    if 'Label' not in colset:
        if 'Attack' in colset:
            uniq = sorted(df['Attack'].astype(str).unique())
            mapping = {a: i for i, a in enumerate(uniq)}
            df['Label'] = df['Attack'].astype(str).map(mapping).astype(int)
        else:
            raise ValueError("Input must contain 'Label' or 'Attack' column")

    # Drop known string identifiers that are not used as numeric flow features
    drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L7_PROTO', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
                 'FTP_COMMAND_RET_CODE', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Replace infinities and NaNs with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Build udps.* columns expected by NIDSDataset
    udps_defaults = {
        'udps.payload_data': "['00']",
        'udps.packet_direction': "['0']",
        'udps.ip_size': "['0']",
        'udps.transport_size': "['0']",
        'udps.payload_size': "['0']",
        'udps.delta_time': "['0']",
    }
    for col, default in udps_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Flags: safe zero defaults
    flag_cols = ['udps.syn', 'udps.cwr', 'udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin']
    for f in flag_cols:
        if f not in df.columns:
            df[f] = "['0']"

    # Ensure flow-level features are numeric
    keep_udps = list(udps_defaults.keys()) + flag_cols
    others = [c for c in df.columns if c not in keep_udps + ['Label']]
    for c in others:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Reorder columns: flow numeric features first, then udps.*, flags, then Label
    udps_cols = list(udps_defaults.keys())
    flow_cols = [c for c in df.columns if c not in udps_cols + flag_cols + ['Label']]
    out_df = df[flow_cols + udps_cols + flag_cols + ['Label']]

    # Optional balancing
    if balance:
        grouped = []
        classes = out_df['Label'].unique()
        for c in classes:
            cls = out_df[out_df['Label'] == c]
            if len(cls) > balance_per_class:
                cls = cls.sample(balance_per_class, random_state=random_state)
            else:
                cls = cls.sample(balance_per_class, replace=True, random_state=random_state)
            grouped.append(cls)
        out_df = pd.concat(grouped).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Stratified split
    strat = out_df['Label'] if out_df['Label'].nunique() > 1 else None
    temp_size = test_size + val_size
    train_df, temp_df = train_test_split(out_df, test_size=temp_size, random_state=random_state, stratify=strat)
    if val_size > 0:
        val_frac_of_temp = val_size / temp_size
        strat_temp = temp_df['Label'] if temp_df['Label'].nunique() > 1 else None
        val_df, test_df = train_test_split(temp_df, test_size=(1 - val_frac_of_temp), random_state=random_state, stratify=strat_temp)
    else:
        val_df = pd.DataFrame(columns=out_df.columns)
        test_df = temp_df

    # Save CSVs
    train_path = os.path.join(out_dir, 'train.csv')
    val_path = os.path.join(out_dir, 'val.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Saved:", train_path, train_df.shape, val_path, val_df.shape, test_path, test_df.shape)
    return train_path, val_path, test_path

if __name__ == '__main__':
    preprocess_and_split(input_csv='data/NF-UNSW-NB15-v4.csv', out_dir='data/processed_unsw')