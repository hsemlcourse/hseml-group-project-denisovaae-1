from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from data_tools import load_csv
from preprocessing import split_by_time

def test_load_raw_csv_smoke():
    raw_path = ROOT / 'data' / 'raw' / 'electricity_load.csv'
    df = load_csv(raw_path)
    assert not df.empty
    assert 'nat_demand' in df.columns
    assert 'datetime' in df.columns

def test_split_by_time_keeps_order_and_sizes():
    df = pd.DataFrame({'x': list(range(100))})
    train_df, val_df, test_df = split_by_time(df, train_frac=0.7, val_frac=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df['x'].iloc[0] == 0
    assert train_df['x'].iloc[-1] == 69
    assert val_df['x'].iloc[0] == 70
    assert test_df['x'].iloc[-1] == 99
