from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_name: str | None = None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(description='Признаки')


class PredictResponse(BaseModel):
    predicted_load: float
    model: str
