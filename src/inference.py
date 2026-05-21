import json

import joblib
import pandas as pd

from data_tools import get_target
from settings import FILE_BEST_MODEL, MODELS_DIR, PROCESSED_VAL_PATH, REPORTS_DIR

FEATURES_PATH = MODELS_DIR / 'deploy_features.json'
META_PATH = REPORTS_DIR / 'best_model_info.json'


def _feature_columns():
    if FEATURES_PATH.is_file():
        return json.loads(FEATURES_PATH.read_text(encoding='utf-8'))
    val = pd.read_csv(PROCESSED_VAL_PATH)
    target = get_target(val)
    drop = {target, 'datetime', 'Datetime', 'date', 'Date', 'timestamp', 'Timestamp'}
    feats = [c for c in val.columns if c not in drop]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text(json.dumps(feats, ensure_ascii=False, indent=2), encoding='utf-8')
    return feats


def load_default_row():
    val = pd.read_csv(PROCESSED_VAL_PATH)
    row = val.iloc[len(val) // 2]
    feats = _feature_columns()
    return {c: float(row[c]) for c in feats}


def load_model():
    if not (MODELS_DIR / FILE_BEST_MODEL).is_file():
        raise FileNotFoundError(f'Нет модели {MODELS_DIR / FILE_BEST_MODEL}')
    return joblib.load(MODELS_DIR / FILE_BEST_MODEL)


def predict_load(features: dict):
    model = load_model()
    cols = _feature_columns()
    row = {c: float(features[c]) for c in cols}
    frame = pd.DataFrame([row])
    pred = float(model.predict(frame)[0])
    meta = {}
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    return {
        'predicted_load': round(pred, 4),
        'model': meta.get('best_model', 'best_model'),
    }
