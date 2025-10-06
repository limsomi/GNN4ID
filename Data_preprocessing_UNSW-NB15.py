import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def to_list_str(v):
    return "['{}']".format(v)

def preprocess_singlefile(input_csv=r'data\\NF-UNSW-NB15-v4.csv',
                          out_dir=r'data\\processed_unsw',
                          test_size=0.2, val_size=0.2,
                          random_state=42, balance=False, balance_per_class=20000):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "shape:", df.shape)

    # Normalize column names to uppercase for robust mapping
    df.columns = [c.strip() for c in df.columns]
    colset = set(df.columns)

    # 1) Ensure Label column exists (prefer 'Label', else map 'Attack')
    if 'Label' not in colset:
        if 'Attack' in colset:
            uniq = sorted(df['Attack'].astype(str).unique())
            mapping = {a: i for i, a in enumerate(uniq)}
            df['Label'] = df['Attack'].astype(str).map(mapping).astype(int)
        else:
            raise ValueError("Input must contain 'Label' or 'Attack' column")

    # 2) Drop known string identifiers that are not used as numeric flow features
    drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L7_PROTO', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
                 'FTP_COMMAND_RET_CODE', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # 3) Basic clean and ensure numeric where applicable
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 4) Build udps.* columns expected by NIDSDataset (stringified single-element lists)
    # payload_data: valid hex -> minimal '00'
    if 'udps.payload_data' not in df.columns:
        df['udps.payload_data'] = "['00']"

    # packet_direction: use IN_PKTS vs OUT_PKTS if present
    if 'udps.packet_direction' not in df.columns:
        if 'IN_PKTS' in df.columns and 'OUT_PKTS' in df.columns:
            df['udps.packet_direction'] = df.apply(
                lambda r: "['0']" if float(r.get('IN_PKTS', 0) or 0) >= float(r.get('OUT_PKTS', 0) or 0) else "['1']",
                axis=1
            )
        else:
            df['udps.packet_direction'] = "['0']"

    # ip_size / transport_size: prefer MIN_IP_PKT_LEN then MAX_IP_PKT_LEN
    for udps_col, src1, src2 in [
        ('udps.ip_size', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN'),
        ('udps.transport_size', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN'),
    ]:
        if udps_col not in df.columns:
            if src1 in df.columns:
                df[udps_col] = df[src1].astype(int).fillna(0).apply(lambda v: f"['{int(v)}']")
            elif src2 in df.columns:
                df[udps_col] = df[src2].astype(int).fillna(0).apply(lambda v: f"['{int(v)}']")
            else:
                df[udps_col] = "['0']"

    # payload_size: prefer IN_BYTES then OUT_BYTES then IN_PKTS
    if 'udps.payload_size' not in df.columns:
        if 'IN_BYTES' in df.columns:
            df['udps.payload_size'] = df['IN_BYTES'].astype(int).fillna(0).apply(lambda v: f"['{int(v)}']")
        elif 'OUT_BYTES' in df.columns:
            df['udps.payload_size'] = df['OUT_BYTES'].astype(int).fillna(0).apply(lambda v: f"['{int(v)}']")
        elif 'IN_PKTS' in df.columns:
            df['udps.payload_size'] = df['IN_PKTS'].astype(int).fillna(0).apply(lambda v: f"['{int(v)}']")
        else:
            df['udps.payload_size'] = "['0']"

    # delta_time: use SRC_TO_DST_IAT_AVG / DST_TO_SRC_IAT_AVG else FLOW_DURATION_MILLISECONDS
    if 'udps.delta_time' not in df.columns:
        if 'SRC_TO_DST_IAT_AVG' in df.columns:
            df['udps.delta_time'] = df['SRC_TO_DST_IAT_AVG'].fillna(0).astype(int).apply(lambda v: f"['{int(v)}']")
        elif 'DST_TO_SRC_IAT_AVG' in df.columns:
            df['udps.delta_time'] = df['DST_TO_SRC_IAT_AVG'].fillna(0).astype(int).apply(lambda v: f"['{int(v)}']")
        elif 'FLOW_DURATION_MILLISECONDS' in df.columns:
            df['udps.delta_time'] = df['FLOW_DURATION_MILLISECONDS'].fillna(0).astype(int).apply(lambda v: f"['{int(v)}']")
        else:
            df['udps.delta_time'] = "['0']"

    # flags: safe zero defaults (NIDSDataset will ignore if include_packetflag=False)
    flag_cols = ['udps.syn','udps.cwr','udps.ece','udps.urg','udps.ack','udps.psh','udps.rst','udps.fin']
    for f in flag_cols:
        if f not in df.columns:
            df[f] = "['0']"

    # 5) Ensure flow-level features are numeric. Keep udps.* & flags & Label as object strings.
    keep_udps = ['udps.payload_data','udps.delta_time','udps.packet_direction','udps.ip_size','udps.transport_size','udps.payload_size'] + flag_cols
    others = [c for c in df.columns if c not in keep_udps + ['Label']]
    for c in others:
        # try convert to numeric, if fail drop
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 6) Reorder columns: flow numeric features first, then udps.*, flags, then Label
    udps_cols = ['udps.payload_data','udps.delta_time','udps.packet_direction','udps.ip_size','udps.transport_size','udps.payload_size']
    flow_cols = [c for c in df.columns if c not in udps_cols + flag_cols + ['Label']]
    out_df = df[flow_cols + udps_cols + flag_cols + ['Label']]

    # 7) Optional balancing
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

    # 8) Stratified split
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

    # 9) Save CSVs
    train_path = os.path.join(out_dir, 'train.csv')
    val_path = os.path.join(out_dir, 'val.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Saved:", train_path, train_df.shape, val_path, val_df.shape, test_path, test_df.shape)
    return train_path, val_path, test_path

if __name__ == '__main__':
    preprocess_singlefile(input_csv=r'data\\NF-UNSW-NB15-v3.csv', out_dir=r'data\\processed_unsw')