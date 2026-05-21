import json

from fastapi import FastAPI, HTTPException

from api.deps import model_ready
from api.schemas import HealthResponse, PredictRequest, PredictResponse
from inference import META_PATH, load_default_row, predict_load

app = FastAPI(
    title='Electricity Load API',
    description='Прогноз почасовой нагрузки',
    version='1.0.0',
)


@app.get('/health', response_model=HealthResponse)
def health():
    name = None
    if model_ready() and META_PATH.is_file():
        name = json.loads(META_PATH.read_text(encoding='utf-8')).get('best_model')
    return HealthResponse(status='ok', model_loaded=model_ready(), model_name=name)


@app.get('/defaults')
def defaults():
    if not model_ready():
        raise HTTPException(status_code=503, detail='Сначала обучите модель: python src/modeling.py')
    return load_default_row()


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    if not model_ready():
        raise HTTPException(status_code=503, detail='Модель не найдена')
    try:
        return PredictResponse(**predict_load(body.features))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f'Нет признака: {exc}') from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
