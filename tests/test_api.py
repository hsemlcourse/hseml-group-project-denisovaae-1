import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from api.app import app
from inference import load_default_row
from settings import FILE_BEST_MODEL, MODELS_DIR

client = TestClient(app)
MODEL_PATH = MODELS_DIR / FILE_BEST_MODEL


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason='python src/modeling.py')
def test_predict():
    payload = {'features': load_default_row()}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body['predicted_load'] > 0
