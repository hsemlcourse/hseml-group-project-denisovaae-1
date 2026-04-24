from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'electricity_load.csv'
PROCESSED_TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train_processed.csv'
PROCESSED_VAL_PATH = PROJECT_ROOT / 'data' / 'processed' / 'val_processed.csv'
PROCESSED_TEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'test_processed.csv'

REPORTS_DIR = PROJECT_ROOT / 'report'
REPORT_IMAGES_DIR = REPORTS_DIR / 'images'
MODELS_DIR = PROJECT_ROOT / 'models'

PLOT_TS = 'eda_load_curve.png'
PLOT_CORR = 'eda_corr_heatmap.png'
PLOT_HOUR = 'eda_hourly_pattern.png'
FILE_SUMMARY = 'data_summary.json'
FILE_EXPERIMENTS = 'experiments.csv'
FILE_BEST_META = 'best_model_info.json'
FILE_BEST_MODEL = 'best_model.joblib'
