from datetime import datetime
import csv

import pandas as pd


DATETIME_CANDIDATES = (
    'datetime',
    'Datetime',
    'date',
    'Date',
    'timestamp',
    'Timestamp',
)

TARGET_CANDIDATES = (
    'nat_demand',
    'demand',
    'Demand',
    'load',
    'Load',
    'target',
    'TARGET',
)


def _detect_dialect(sample_text):
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=',;\t|')
    except csv.Error:
        return csv.get_dialect('excel')


def _parse_datetime(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S', '%d.%m.%Y %H:%M'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors='coerce')
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def parse_csv_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')

    raw_text = path.read_text(encoding='utf-8', errors='ignore')
    sample = raw_text[:10000]
    dialect = _detect_dialect(sample)

    rows = []
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as fobj:
        reader = csv.DictReader(fobj, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError('empty header in source dataset')
        for src_row in reader:
            clean_row = {str(key).strip(): (value.strip() if isinstance(value, str) else value) for key, value in src_row.items()}
            rows.append(clean_row)

    if not rows:
        raise ValueError('empty source dataset')

    frame = pd.DataFrame(rows)

    dt_col = next((col for col in DATETIME_CANDIDATES if col in frame.columns), None)
    if dt_col is not None:
        frame[dt_col] = frame[dt_col].map(_parse_datetime)

    target_col = next((col for col in TARGET_CANDIDATES if col in frame.columns), None)
    if target_col is not None:
        frame[target_col] = pd.to_numeric(frame[target_col], errors='coerce')

    for col in frame.columns:
        if col == dt_col:
            continue
        if frame[col].dtype == object:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except (ValueError, TypeError):
                pass

    return frame
