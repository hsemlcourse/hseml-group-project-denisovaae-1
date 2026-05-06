import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from data_tools import get_target
from settings import (
    FILE_BEST_META,
    FILE_BEST_MODEL,
    FILE_EXPERIMENTS,
    MODELS_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)


def calc_metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_models():
    return {
        'baseline_linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'knn': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=8))]),
        'rf_200': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'xgb_small': XGBRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'xgboost': XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_tuned_models(x_train, y_train):
    tscv = TimeSeriesSplit(n_splits=4)
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions={
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 14, 18, None],
            'min_samples_split': [2, 4, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', None],
        },
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(x_train, y_train)

    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective='reg:squarederror',
        ),
        param_distributions={
            'n_estimators': [180, 260, 340],
            'learning_rate': [0.04, 0.06, 0.08],
            'max_depth': [4, 5, 6],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_lambda': [1.0, 3.0, 6.0],
            'min_child_weight': [1, 3, 5],
        },
        n_iter=12,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_search.fit(x_train, y_train)

    return {
        'rf_random_search': rf_search.best_estimator_,
        'xgb_random_search': xgb_search.best_estimator_,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_table = pd.read_csv(PROCESSED_TRAIN_PATH)
    valid_table = pd.read_csv(PROCESSED_VAL_PATH)
    test_table = pd.read_csv(PROCESSED_TEST_PATH)
    target_col = get_target(train_table)

    drop = [target_col, 'datetime', 'Datetime', 'date', 'Date', 'timestamp', 'Timestamp']
    feats = [i for i in train_table.columns if i not in drop]
    base_drop = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_24', 'rolling_mean_24']
    raw_feats = [i for i in feats if i not in base_drop]

    x_train, y_train = train_table[feats], train_table[target_col]
    x_valid, y_valid = valid_table[feats], valid_table[target_col]
    x_test, y_test = test_table[feats], test_table[target_col]
    x_train_raw = train_table[raw_feats]
    x_valid_raw = valid_table[raw_feats]
    x_test_raw = test_table[raw_feats]

    rows = []
    models = build_models()
    models.update(build_tuned_models(x_train, y_train))
    best_model_name = ''
    best_valid_mae = float('inf')

    baseline_raw = LinearRegression()
    baseline_raw.fit(x_train_raw, y_train)
    raw_valid_pred = baseline_raw.predict(x_valid_raw)
    raw_test_pred = baseline_raw.predict(x_test_raw)
    raw_valid_metrics = calc_metrics(y_valid, raw_valid_pred)
    raw_test_metrics = calc_metrics(y_test, raw_test_pred)
    rows.append(
        {
            'model': 'baseline_raw_linear',
            'feature_set': 'raw_only',
            'val_mae': raw_valid_metrics['mae'],
            'val_rmse': raw_valid_metrics['rmse'],
            'test_mae': raw_test_metrics['mae'],
            'test_rmse': raw_test_metrics['rmse'],
        }
    )

    for i, model in models.items():
        model.fit(x_train, y_train)
        valid_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        valid_metrics = calc_metrics(y_valid, valid_pred)
        test_metrics = calc_metrics(y_test, test_pred)
        rows.append(
            {
                'model': i,
                'feature_set': 'full_features',
                'val_mae': valid_metrics['mae'],
                'val_rmse': valid_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
            }
        )
        if valid_metrics['mae'] < best_valid_mae:
            best_valid_mae = valid_metrics['mae']
            best_model_name = i
            joblib.dump(model, MODELS_DIR / FILE_BEST_MODEL)

    results_df = pd.DataFrame(rows).sort_values(by='val_mae')
    results_df.to_csv(REPORTS_DIR / FILE_EXPERIMENTS, index=False)
    (REPORTS_DIR / FILE_BEST_META).write_text(
        json.dumps({'best_model': best_model_name, 'selection_metric': 'val_mae'}, indent=2),
        encoding='utf-8',
    )
    print(results_df)


if __name__ == '__main__':
    main()
