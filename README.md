# House Price Prediction

ML pipeline for predicting real estate prices with model versioning and REST API deployment.

## Project Overview

- **Dataset**: 247K real estate transactions (2020-2024)
- **Best Model**: Stacking Ensemble (R² = 0.98, RMSE = $28.5K, MAE = $18.5K)
- **Features**: 26 engineered features, multi-core processing, model versioning, REST API
- **Deployment**: CI/CD via GitHub → Render.com

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run training
python main.py --train

# Start API server
python main.py --api

# For EDA, open the notebook
jupyter notebook notebooks/eda_and_feature_engineering.ipynb
```

## Project Structure

```text
house_price_prediction/
├── .github/workflows/
│   └── deploy.yml              # CI/CD pipeline
├── docs/
│   ├── feature_importance_chart.png
│   ├── feature_engineering_chart.png
│   └── Case Study 1 Data (1).xlsx
├── notebooks/
│   └── eda_and_feature_engineering.ipynb
├── src/
│   ├── preprocessing.py        # Data preprocessing (Preprocessor + ImprovedPreprocessor)
│   ├── model_training.py       # Model training with stacking ensemble
│   └── model_registry.py       # Version management
├── api/
│   └── app.py                  # FastAPI application
├── tests/
│   ├── test_api.py             # API tests
│   └── load_test.js            # k6 load tests
├── models/
│   ├── feature_importance.csv  # Feature importance scores
│   ├── model_comparison.csv    # Model performance comparison
│   ├── test_predictions.csv    # Test set predictions
│   └── registry/               # Versioned models
│       ├── manifest.json
│       └── house_price_predictor/
│           ├── v1.0.0/         # LightGBM baseline (MAE: $117K)
│           ├── v1.0.1/         # LightGBM + features (MAE: $115K)
│           └── v1.1.0/         # Stacking Ensemble (MAE: $18.5K) ← production
├── Dockerfile                  # Container config
├── render.yaml                 # Render.com config
├── main.py
└── requirements.txt
```

## Models Evaluated

| Model | RMSE | MAE | R² | Training Time |
|-------|------|-----|-----|---------------|
| **Stacking Ensemble** | **$28,525** | **$18,470** | **0.9846** | 585s |
| XGBoost | $28,990 | $18,756 | 0.9841 | 2.2s |
| LightGBM | $28,892 | $18,870 | 0.9842 | 5.5s |
| Gradient Boosting | $29,098 | $19,086 | 0.9840 | 207s |
| Random Forest | $31,901 | $21,703 | 0.9807 | 12s |

The Stacking Ensemble combines LightGBM, XGBoost, GradientBoosting, and RandomForest with a Ridge meta-learner.

## Top Predictive Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `days_since_2020` | 728 | Temporal |
| 2 | `Location_encoded` | 621 | Encoded |
| 3 | `size_per_room` | 579 | Size |
| 4 | `Condition_encoded` | 566 | Encoded |
| 5 | `size_time_interaction` | 510 | Interaction |
| 6 | `Size` | 478 | Size |
| 7 | `size_per_bedroom` | 403 | Size |
| 8 | `day_of_year` | 358 | Temporal |
| 9 | `log_size` | 281 | Size |
| 10 | `property_age` | 247 | Age |

## Feature Engineering (26 features)

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** (7) | `days_since_2020`, `year_sold`, `month_sold`, `quarter`, `day_of_year`, `year_sold_numeric`, `is_spring_summer` | Time-based features capturing market trends |
| **Size** (5) | `Size`, `size_squared`, `log_size`, `size_per_bedroom`, `size_per_room` | Property size and derived metrics |
| **Room** (5) | `Bedrooms`, `Bathrooms`, `total_rooms`, `bath_ratio`, `bed_bath_product` | Room counts and ratios |
| **Age** (4) | `property_age`, `decade_built`, `is_new_construction`, `is_recent` | Property age features |
| **Interactions** (2) | `size_time_interaction`, `age_size_interaction` | Cross-feature interactions |
| **Encoded** (3) | `Location_encoded`, `Condition_encoded`, `Type_encoded` | Target-encoded categoricals |

## Key Improvements (v1.1.0)

- **Target Encoding**: Replaced one-hot encoding with target encoding for categorical features
- **Time-based Features**: Added `size_time_interaction` to capture price appreciation with property size
- **Stacking Ensemble**: Combined 4 base models with Ridge meta-learner for better generalization
- **Result**: MAE reduced from $115K to $18.5K (84% improvement)

## REST API

**Live API:** https://agent-mira-assignment-data-scientist.onrender.com

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single property prediction |
| `/predict/batch` | POST | Batch predictions |
| `/health` | GET | Health check |
| `/model/info` | GET | Model details & metrics |
| `/model/versions` | GET | List all versions |
| `/model/load/{version}` | POST | Load specific version |
| `/model/load?stage=production` | POST | Load by stage |
| `/model/compare?v1=v1.0.0&v2=v1.1.0` | GET | Compare versions |

**Example Request:**

```bash
curl -X POST "https://agent-mira-assignment-data-scientist.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "CityA",
    "size": 2500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2010,
    "condition": "Good",
    "property_type": "Single Family"
  }'
```

**Response:**
```json
{
  "predicted_price": 448825.88,
  "price_low": 403943.29,
  "price_high": 493708.47,
  "model_version": "v1.1.0",
  "details": {
    "location": "CityA",
    "size": 2500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2010,
    "condition": "Good",
    "type": "Single Family"
  }
}
```

**Interactive Documentation:** https://agent-mira-assignment-data-scientist.onrender.com/docs

## Model Versioning

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry("models/registry")

# Register new version
version = registry.register(
    model=trained_model,
    model_name="house_price_predictor",
    metrics={"rmse": 28525, "mae": 18470, "r2": 0.9846},
    params={"model_type": "StackingRegressor"},
    tags=["stacking", "ensemble"]
)

# Promote to production
registry.promote("house_price_predictor", version, stage="production")

# Load model
model, preprocessor, metadata = registry.load("house_price_predictor", stage="production")

# Compare versions
registry.compare("house_price_predictor", "v1.0.1", "v1.1.0")

# View all versions
registry.summary()
```

## CI/CD Deployment

**Pipeline:** GitHub → Render.com → Docker → Live API

1. Push code to GitHub (`git push origin main`)
2. Render.com auto-detects changes via webhook
3. Docker builds image using `Dockerfile`
4. API deploys automatically

**Live URL:** https://agent-mira-assignment-data-scientist.onrender.com

## Load Testing

Run load tests using k6:

```bash
k6 run tests/load_test.js
```

**Results (10 concurrent users, 40s duration):**

| Metric | Value |
|--------|-------|
| Total Requests | 300 |
| Success Rate | 100% |
| Requests/sec | 7.34 |
| Avg Response Time | 503ms |
| p95 Response Time | 991ms |

## Test Predictions Summary

| Metric | Value |
|--------|-------|
| Test Samples | 48,347 |
| MAE | $18,470 |
| Median Error | $14,254 |
| 95th Percentile | $44,341 |

## Requirements

- Python 3.9+
- pandas, numpy, openpyxl
- scikit-learn, xgboost, lightgbm
- fastapi, uvicorn, pydantic
- joblib, matplotlib

## License

MIT
