from pathlib import Path

import numpy as np
import pandas as pd


def get_target(df):
    cols_try = ['nat_demand', 'load', 'Load', 'target', 'TARGET', 'demand', 'Demand']
    for i in cols_try:
        if i in df.columns:
            return i

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    demand_like = [i for i in numeric_cols if ('demand' in i.lower() or 'load' in i.lower())]
    if demand_like:
        return demand_like[0]

    stable_numeric = [i for i in numeric_cols if df[i].nunique(dropna=True) > 20]
    if stable_numeric:
        return stable_numeric[0]

    if not numeric_cols:
        raise ValueError('no numeric columns')
    return numeric_cols[0]


def get_datetime(df):
    cols_try = ['datetime', 'Datetime', 'date', 'Date', 'timestamp', 'Timestamp']
    for i in cols_try:
        if i in df.columns:
            return i

    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not object_cols:
        raise ValueError('no datetime-like column')
    return object_cols[0]


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')
    return pd.read_csv(path)
