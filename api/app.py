"""
FastAPI app for house price prediction with model versioning
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import pandas as pd
from pathlib import Path
from datetime import date
import uvicorn
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="ML API with model versioning"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).parent.parent / "models"

# globals
model = None
preprocessor = None
model_metadata = None
registry = None
executor = None


class PropertyInput(BaseModel):
    location: str = Field(..., description="City name")
    size: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=5)
    year_built: int = Field(..., ge=1800, le=2025)
    condition: str
    property_type: str
    date_sold: Optional[date] = None

    @field_validator('condition')
    @classmethod
    def check_condition(cls, v):
        valid = ['New', 'Good', 'Fair', 'Poor']
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    @field_validator('property_type')
    @classmethod
    def check_type(cls, v):
        valid = ['Single Family', 'Condominium', 'Townhouse']
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "CityA",
                "size": 2500,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2010,
                "condition": "Good",
                "property_type": "Single Family"
            }
        }
    }


class BatchInput(BaseModel):
    properties: List[PropertyInput]


class PredictionResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    model_version: str
    details: dict


def to_dataframe(prop: PropertyInput):
    sale_date = prop.date_sold or date.today()
    return pd.DataFrame([{
        'Property ID': 'API',
        'Location': prop.location,
        'Size': prop.size,
        'Bedrooms': prop.bedrooms,
        'Bathrooms': prop.bathrooms,
        'Year Built': prop.year_built,
        'Condition': prop.condition,
        'Type': prop.property_type,
        'Date Sold': pd.Timestamp(sale_date)
    }])


def predict_one(df):
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    margin = pred * 0.1
    return pred, pred - margin, pred + margin


def load_model(version=None, stage="production"):
    """Load model from registry"""
    global model, preprocessor, model_metadata, registry

    from model_registry import ModelRegistry
    from preprocessing import Preprocessor

    if registry is None:
        registry = ModelRegistry(str(MODEL_DIR / "registry"))

    if version:
        loaded_model, prep_data, metadata = registry.load("house_price_predictor", version=version)
    else:
        loaded_model, prep_data, metadata = registry.load("house_price_predictor", stage=stage)

    model = loaded_model
    model_metadata = metadata

    preprocessor = Preprocessor()
    preprocessor.encoders = prep_data['encoders']
    preprocessor.scaler = prep_data['scaler']
    preprocessor.feature_cols = prep_data['feature_cols']
    preprocessor.num_cols = prep_data['num_cols']
    preprocessor.cat_cols = prep_data['cat_cols']

    return True


@app.on_event("startup")
async def startup():
    global executor

    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"MODEL_DIR exists: {MODEL_DIR.exists()}")
    registry_path = MODEL_DIR / "registry"
    print(f"Registry path: {registry_path}")
    print(f"Registry exists: {registry_path.exists()}")

    if registry_path.exists():
        import os
        print(f"Registry contents: {os.listdir(registry_path)}")

    try:
        load_model(stage="production")
        print(f"Loaded model: {model_metadata.get('version')} [{model_metadata.get('stage')}]")
    except Exception as e:
        import traceback
        print(f"Warning: could not load model - {e}")
        traceback.print_exc()
        print("Run training first: python main.py --train")

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    executor = ThreadPoolExecutor(max_workers=n_workers)


@app.on_event("shutdown")
async def shutdown():
    if executor:
        executor.shutdown(wait=True)


@app.get("/")
async def root():
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "model_version": model_metadata.get("version") if model_metadata else None
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "no_model",
        "model_loaded": model is not None,
        "model_version": model_metadata.get("version") if model_metadata else None,
        "model_stage": model_metadata.get("stage") if model_metadata else None,
        "cores": multiprocessing.cpu_count()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(prop: PropertyInput):
    if not model:
        raise HTTPException(503, "Model not loaded. Run training first.")

    try:
        df = to_dataframe(prop)
        pred, low, high = predict_one(df)

        return PredictionResponse(
            predicted_price=round(pred, 2),
            price_low=round(low, 2),
            price_high=round(high, 2),
            model_version=model_metadata.get("version", "unknown"),
            details={
                "location": prop.location,
                "size": prop.size,
                "bedrooms": prop.bedrooms,
                "bathrooms": prop.bathrooms,
                "year_built": prop.year_built,
                "condition": prop.condition,
                "type": prop.property_type
            }
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    if not model:
        raise HTTPException(503, "Model not loaded")

    loop = asyncio.get_event_loop()
    results = []

    for prop in batch.properties:
        df = to_dataframe(prop)
        pred, low, high = await loop.run_in_executor(executor, predict_one, df)
        results.append({
            "predicted_price": round(pred, 2),
            "price_low": round(low, 2),
            "price_high": round(high, 2)
        })

    return {
        "predictions": results,
        "count": len(results),
        "model_version": model_metadata.get("version", "unknown")
    }


@app.get("/model/info")
async def get_model_info():
    if not model:
        raise HTTPException(503, "Model not loaded")

    info = {
        "type": type(model).__name__,
        "version": model_metadata.get("version"),
        "stage": model_metadata.get("stage"),
        "metrics": model_metadata.get("metrics"),
        "params": model_metadata.get("params"),
        "tags": model_metadata.get("tags"),
        "features": preprocessor.feature_cols if preprocessor else None
    }

    if hasattr(model, 'feature_importances_') and preprocessor:
        imp = dict(zip(preprocessor.feature_cols, model.feature_importances_.tolist()))
        info['top_features'] = dict(sorted(imp.items(), key=lambda x: -x[1])[:5])

    return info


@app.get("/model/versions")
async def list_versions():
    if not registry:
        raise HTTPException(503, "Registry not initialized")

    versions = registry.list_versions("house_price_predictor")
    return {
        "model_name": "house_price_predictor",
        "versions": versions,
        "current": model_metadata.get("version") if model_metadata else None
    }


@app.post("/model/load/{version}")
async def load_version(version: str):
    try:
        load_model(version=version)
        return {
            "message": f"Loaded version {version}",
            "version": model_metadata.get("version"),
            "metrics": model_metadata.get("metrics")
        }
    except Exception as e:
        raise HTTPException(404, str(e))


@app.post("/model/load")
async def load_by_stage(stage: str = Query("production", enum=["production", "staging", "development"])):
    try:
        load_model(stage=stage)
        return {
            "message": f"Loaded {stage} model",
            "version": model_metadata.get("version"),
            "stage": stage
        }
    except Exception as e:
        raise HTTPException(404, str(e))


@app.get("/model/compare")
async def compare_versions(v1: str, v2: str):
    if not registry:
        raise HTTPException(503, "Registry not initialized")

    try:
        m1, m2 = registry.compare("house_price_predictor", v1, v2)
        return {
            "version_1": {"version": v1, "metrics": m1["metrics"]},
            "version_2": {"version": v2, "metrics": m2["metrics"]},
            "diff": {
                k: round(m2["metrics"].get(k, 0) - m1["metrics"].get(k, 0), 4)
                for k in m1["metrics"]
            }
        }
    except Exception as e:
        raise HTTPException(404, str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
