import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_tools import get_datetime, get_target, load_csv
from settings import (
    FILE_SUMMARY,
    PLOT_CORR,
    PLOT_HOUR,
    PLOT_TS,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    RAW_DATA_PATH,
    REPORT_IMAGES_DIR,
    REPORTS_DIR,
)


def add_time_features(raw_table, datetime_col):
    data = raw_table.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
    data = data.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    data['hour'] = data[datetime_col].dt.hour
    data['day_of_week'] = data[datetime_col].dt.dayofweek
    data['month'] = data[datetime_col].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    return data


def add_lag_features(clean_table, target_col):
    data = clean_table.copy()
    base = data[target_col]
    data['lag_1'] = base.shift(1)
    data['lag_24'] = base.shift(24)
    data['rolling_mean_24'] = base.shift(1).rolling(24).mean()
    return data.dropna().reset_index(drop=True)


def cut_outliers_iqr(clean_table, target_col):
    q1 = clean_table[target_col].quantile(0.25)
    q3 = clean_table[target_col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return clean_table[(clean_table[target_col] >= low) & (clean_table[target_col] <= high)].copy()


def split_by_time(full_table, train_frac=0.7, val_frac=0.15):
    train_end = int(len(full_table) * train_frac)
    val_end = int(len(full_table) * (train_frac + val_frac))
    return (
        full_table.iloc[:train_end].copy(),
        full_table.iloc[train_end:val_end].copy(),
        full_table.iloc[val_end:].copy(),
    )


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_csv(RAW_DATA_PATH)
    initial_shape = raw_df.shape
    dup_count = int(raw_df.duplicated().sum())

    datetime_col = get_datetime(raw_df)
    target_col = get_target(raw_df)
    work_df = raw_df.drop_duplicates().copy()
    work_df = add_time_features(work_df, datetime_col=datetime_col)
    work_df[target_col] = pd.to_numeric(work_df[target_col], errors='coerce')
    miss_target = int(work_df[target_col].isna().sum())
    work_df = work_df.dropna(subset=[target_col]).copy()
    before_out = len(work_df)
    work_df = cut_outliers_iqr(work_df, target_col=target_col)
    out_removed = int(before_out - len(work_df))
    work_df = add_lag_features(work_df, target_col=target_col)

    train_df, valid_df, test_df = split_by_time(work_df)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    valid_df.to_csv(PROCESSED_VAL_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(work_df[datetime_col].iloc[:2400], work_df[target_col].iloc[:2400], linewidth=1.2)
    plt.title('Нагрузка по часам (начало периода)')
    plt.xlabel('Datetime')
    plt.ylabel('Load')
    plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / PLOT_TS, dpi=150)
    plt.close()

    corr = work_df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 8))
    plt.imshow(corr, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Матрица корреляций признаков')
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / PLOT_CORR, dpi=150)
    plt.close()

    hourly_profile = work_df.groupby('hour')[target_col].mean()
    plt.figure(figsize=(10, 4))
    plt.plot(hourly_profile.index, hourly_profile.values, marker='o')
    plt.title('Средняя нагрузка по часам суток')
    plt.xlabel('Hour')
    plt.ylabel('Mean load')
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / PLOT_HOUR, dpi=150)
    plt.close()

    new_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_24', 'rolling_mean_24']
    summary = {
        'initial_rows': int(initial_shape[0]),
        'initial_columns': int(initial_shape[1]),
        'final_rows': int(work_df.shape[0]),
        'final_columns': int(work_df.shape[1]),
        'duplicates_removed': dup_count,
        'missing_target_rows_removed': miss_target,
        'outliers_removed': out_removed,
        'datetime_column': datetime_col,
        'target_column': target_col,
        'added_features': new_cols,
        'train_rows': int(train_df.shape[0]),
        'val_rows': int(valid_df.shape[0]),
        'test_rows': int(test_df.shape[0]),
    }
    (REPORTS_DIR / FILE_SUMMARY).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print('Подготовка завершена')


if __name__ == '__main__':
    main()
